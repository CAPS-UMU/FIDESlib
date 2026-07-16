# CPU/GPU Bit-Compatibility: Issues and Plan

**Goal:** every FIDESlib GPU operation produces ciphertexts that are bit-identical to the
patched-OpenFHE CPU equivalent, verified by exact ciphertext comparison (`ASSERT_EQ_CIPHERTEXT`),
not decrypt-level tolerance. Exact plaintext-level comparison is impossible by design: OpenFHE
adds fresh Gaussian noise to every CKKS decryption (`CKKSPackedEncoding::Decode` noise-flooding
countermeasure), so decrypted values are compared within precision (`ASSERT_ERROR_OK`) only.

**Landing zones.** Changes live in one of three places, in order of preference:

1. **OpenFHE dev branch** — algorithmic changes with standalone value (formulation unification,
   optimizations that also help the CPU).
2. **`deps/fideslib-ref-1.5.1.2.patch`** — kept minimal; now back to visibility/API shims only
   (applies on tag `fideslib-ref-v1.5.1.2`). Temporary home for changes queued for upstream.
3. **FIDESlib** — GPU-side fixes and anything that is a FIDESlib bug.

## Current test status (`test/OpenFheCompatTests.cu`)

| Test | Status | Notes |
|---|---|---|
| `EvalFastRotation` | PASS | single-index hoisted rotation |
| `EvalRotate` | PASS | keyswitch-first path unified onto the hoisted core upstream (O1, `fideslib-ref-v1.5.1.2`) |
| `EvalBootstrap` | PASS | sparse, N=4096, slots=8, FLEXIBLEAUTO, budget {3,3} |
| `EvalBootstrapDense` | PASS | fully packed, slots=N/2 |
| `EvalArithmetic` | PASS | add/sub/negate/mult/square, ct∘ct and ct∘scalar |
| `EvalArithmeticPt` | PASS | ct∘plaintext variants |
| `EvalAdjust` | PASS | 9 mixed noise-degree / level-gap adjustment cases |
| `EvalChebyshev` | PASS | standalone degree-12 Paterson–Stockmeyer |
| `EvalAddMany` | PASS | |
| `DISABLED_EvalFastRotationHoisted` | **disabled** | pre-existing memory bug (issue O2) |

## Resolved issues

### R1. `EvalRotate`/`EvalAtIndex` vs `EvalFastRotation` are different formulations
The hoisted formulation (`EvalFastRotation` HYBRID branch: extended key-switch → fold `c0·P` →
automorph → `ApproxModDown`) and the KeySwitch-first formulation (`EvalRotate`: key-switch,
mod-down, add to `c0`, automorph last) produce different-but-equally-valid ciphertexts — the
key-switching noise enters automorphed in one and un-automorphed in the other, so they never
match bitwise. FIDESlib implements only the hoisted flavor (verified: GPU `rotate` == CPU
`EvalFastRotation` exactly, incl. at 26 towers).
**Interim fix (was in the patch):** the bootstrap's 4 `EvalRotate`/`EvalAtIndex` sites
(PartialSum loop, final sparse doubling, CtS/StC correction rotations) switched to
`EvalFastRotation`; `FHECKKSRNS::Conjugate` rewritten in the hoisted form with
`autoIndex = 2N−1`.
**Superseded by O1:** `EvalRotate`/`EvalAtIndex`/`Conjugate` are now routed through the hoisted
core upstream (`fideslib-ref-v1.5.1.2`), so the original bootstrap call sites are bit-compatible
as written — the five interim hunks were reverted to stock upstream code and dropped from the
patch (they were never upstreamed as call-site changes).

### R2. `Accumulate` lazy extended-basis accumulation *(deliberate GPU optimization, reverted — restore via P2b)*
The GPU PartialSum kept `c0` in the Q·P basis across all doubling levels with one deferred
mod-down. `ApproxModDown` rounds, so `moddown(Σx) ≠ Σ moddown(x)`; OpenFHE's per-rotation
mod-down can never match. **Current fix (FIDESlib `AccumulateBroadcast.cu`):** sequential full
rotations, per-rotation mod-down. Perf cost ≈ 7 extra `c0` mod-downs per bootstrap at the raised
level (the hoisting itself was moot: bootstrap uses `accumulate_bStep = 2`, one rotation/level).

```diff
 	int logbStep = std::bit_width((uint32_t)bStep) - 1;
 	for (int s = 1; s < size; s <<= logbStep) {
-		std::vector<int> indexes;
-		std::vector<Ciphertext*> auxptr;
+		int n = 0;
 		for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
-			indexes.push_back(idx);
-			auxptr.emplace_back(&aux[idx / stride / s - 1]);
+			aux[n].rotate(ctxt, idx);   // full rotation: key-switch + moddown of both components
+			++n;
 		}
-		ctxt.rotate_hoisted(indexes, auxptr, true);  // rotations stay in the extended basis
-		for (size_t i = 0; i < indexes.size(); ++i) {
-			ctxt.add(*auxptr[i]);                    // c0 accumulates extended
+		for (int i = 0; i < n; ++i) {
+			ctxt.add(aux[i]);                        // accumulate in the standard basis
 		}
-		ctxt.c1.moddown();                           // c1 settled per level ...
 	}
-	if (ctxt.c0.isModUp())
-		ctxt.c0.moddown();                           // ... c0 settled ONCE at the end
```

### R3. `LinearTransform` `ONLY_C1` lazy mod-down *(deliberate GPU optimization, reverted — restore via P2c)*
Horner giant-step and correction rotations mod-downed only `c1`, keeping `c0` extended; CPU
`EvalHornerGiantRotate` does a full `KeySwitchDown` (with exact `·PModq` re-lift) per giant step.
**Current fix (FIDESlib `LinearTransform.cu`):** `ONLY_C1 = false`. Perf cost ≈ 1 extra `c0`
mod-down per giant step per CtS/StC level.

What the OpenFHE-side change (P2c) would look like — illustrative sketch of a lazy
`EvalHornerGiantRotate` (`ckksrns-fhe.cpp`); on merge, FIDESlib flips `ONLY_C1` back to `true`:

```diff
 Ciphertext<DCRTPoly> FHECKKSRNS::EvalHornerGiantRotate(ConstCiphertext<DCRTPoly> outer, ...) {
     ...
-    auto outerStd       = cc->KeySwitchDown(outer);   // settles BOTH components (rounds c0)
-    const auto& cvStd   = outerStd->GetElements();
-    const auto paramsQl = cvStd[0].GetParams();
+    // Lazy: settle only c1 (needed for the digit decomposition); c0 stays in the
+    // extended basis across the whole Horner accumulation.
+    const auto& cvExt = outer->GetElements();
+    DCRTPoly c1Std    = cvExt[1].ApproxModDown(paramsQl, cryptoParams->GetParamsP(), ...);
 
-    auto digits = algo->EvalKeySwitchPrecomputeCore(cvStd[1], cryptoParams);
+    auto digits = algo->EvalKeySwitchPrecomputeCore(c1Std, cryptoParams);
     auto cTilda = algo->EvalFastKeySwitchCoreExt(digits, giantKey, paramsQl);
 
-    // addFirst: lift element 0 into the extended basis (c0 * P) and fold it in.
-    DCRTPoly psiC0(cTilda[0].GetParams(), Format::EVALUATION, true);
-    auto cMult            = cvStd[0].TimesNoCheck(cryptoParams->GetPModq());
-    const uint32_t sizeQl = paramsQl->GetParams().size();
-    for (uint32_t i = 0; i < sizeQl; ++i)
-        psiC0.SetElementAtIndex(i, std::move(cMult.GetElementAtIndex(i)));
-    cTilda[0] += psiC0;
+    // c0 is already extended: fold it in directly, skipping the moddown + ·PModq
+    // re-lift round-trip (that round-trip rounds c0 on every giant step).
+    cTilda[0] += cvExt[0];
 
     cTilda[0] = cTilda[0].AutomorphismTransform(autoIndex, map);
     cTilda[1] = cTilda[1].AutomorphismTransform(autoIndex, map);
     ...
 }
```

The level-end `KeySwitchDown` in `EvalCoeffsToSlots`/`EvalSlotsToCoeffs` is unchanged and settles
`c0` once per level. Caveat (same class as P2b): `cTilda[0] += cvExt[0]` assumes the incoming
extended `c0` uses the same ·P-scaled convention as the key-switch product, which must match
FIDESlib's representation exactly — validate with one stage-harness cycle before trusting it.

### R4. GPU Chebyshev was a stale port (values wrong, not just slower)
`evalChebyshevSeries`/inner PS ported an older OpenFHE: mutated shared powers `T[i]` in place
(sticky adjustments), skipped the `AdjustLevelsAndDepthInPlace` power alignment, and the inner
recursion didn't match the current qu/su/cu construction. **Fix (FIDESlib `ApproxModEval.cu`):**
line-for-line transcription of current OpenFHE (clone-based operand adjustment, power alignment,
transcribed inner PS + `EvalPartialLinearWSum`). Locked in by `EvalChebyshev` + both bootstrap tests.

### R5. `adjustScaleAndLevel` deg2→deg2 branch mismatched `AdjustLevelsAndDepthInPlace`
Three separate deltas in `Ciphertext.cpp`: floating-point evaluation order of the adjustment
factor (`scf2*q1/scf1/scf` vs OpenFHE's `scf2/scf1*q1/scf` — differs in ULPs, changes the encoded
scalar), the mod-reduce factor index (target+1 instead of the current top level), and
drop-before-rescale instead of rescale-before-drop. Locked in by `EvalAdjust`.

### R6. `LevelReduceInPlace` is a no-op outside FIXEDMANUAL
OpenFHE only drops levels under FIXEDMANUAL; the GPU inner PS transcription initially dropped a
real level for FLEXIBLE modes. Fixed with a FIXEDMANUAL gate (`su`/`cu` handling in inner PS).

### R7. Double-angle scalar floating-point form
Must be `-pow(2π, -2^i)`; `-1.0/pow(2π, 2^i)` is not FP-identical. Fixed in
`applyDoubleAngleIterations`.

### R8. Missing/misplaced deg-2 rescale before Chebyshev
CPU `EvalBootstrap` runs `ModReduceInternalInPlace` (deg2→deg1) on the ciphertext(s) before
`EvalChebyshevSeries`. Mirrored in `Bootstrap.cu` for the sparse branch and the dense branch
(both `ctxtEnc` and `ctxtEncI`).

### R9. api `EvalNegate` implemented as `multScalar(-1.0)`
Bumped noise degree to 2 and consumed scale — never equivalent to OpenFHE's exact negation, and
silently cost callers a level. **Fix (`api/CryptoContext.cpp`):** exact per-limb multiply by
`q_i − 1`; no metadata change. Applies to `EvalNegate` and `EvalNegateInPlace`.

### R10. api `bad any_cast` in ct×pt CPU fallbacks
`EvalMult(ct, pt)` / `EvalMultInPlace(ct, pt)` cast `pt->cpu` to `ConstPlaintext` where it holds
`Plaintext` — an `std::any` cast must name the stored type exactly, so any CPU-path ct×pt
multiply threw. Fix in `api/CryptoContext.cpp` (both call sites):

```diff
-		auto& ptImpl  = std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
+		auto& ptImpl  = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
```

This failure mode is inherent to the `std::any` design — see O7.

### O1. `EvalRotate` unification *(resolved — landed as `fideslib-ref-v1.5.1.2`)*
The hoisted HYBRID formulation (already used by `EvalFastRotation`, added upstream for the GPU
backend) is extended to `EvalRotate`/`EvalAtIndex`/`EvalAutomorphism`/`Conjugate`, making
`EvalRotate(ct, i)` ≡ `EvalFastRotation(ct, i, m, EvalFastRotationPrecompute(ct))` bit-for-bit:
`EvalAutomorphism` and `EvalFastRotation` both delegate to a new `EvalAutomorphismCore`
(`base-leveledshe.cpp`) holding the former `EvalFastRotation` body, and `FHECKKSRNS::Conjugate`
delegates to `EvalAutomorphism(ct, 2N−1)`. Side effect: upstream `EvalRotate` output bit-patterns
change (same decrypted values and noise magnitude). Landed as commit `eb3e76cf` on branch `gpu`,
tagged `fideslib-ref-v1.5.1.2`; the patch rebased onto it (R1 hunks dropped).
`OpenFHECompatTests.EvalRotate` was the acceptance test and is green, and the api
`AccumulateSum` CPU fallback (an `EvalRotate`+add doubling loop) and `EvalRotateInPlace` are
bit-compatible with it.

## Open issues

### O2. Heap corruption in multi-index `EvalFastRotation` (GPU `rotate_hoisted`, ext=false)
Pre-existing memory bug, **not** a parity issue: rotation values are bit-exact (each result
individually passes the exact comparison; decrypt-level comparison of both passes), but running
the comparison/sync loop over both results crashes with a clobbered `CiphertextImpl` (garbage
jump through the `std::any` manager pointer; `need_lazy_copy` also corrupted). Layout-dependent.
Reproducer: `DISABLED_EvalFastRotationHoisted`.

**Debugging constraint:** our dev environment is WSL2, where `compute-sanitizer` does not run
(WDDM path, "device not supported") — GPU-side memory checking is unavailable to us. Host-side
tools (gdb backtraces/watchpoints, a host ASAN build) do work under WSL and are worth one attempt,
but if the corrupting write originates in a device kernel or an async D2H copy, localizing it
requires native-Linux GPU access. **→ Hand off to the FIDESlib maintainers**: run the reproducer
under `compute-sanitizer --tool memcheck` on a native Linux box; the disabled test plus the notes
above should make it a short investigation.

### O3. api `Rescale` semantic divergence under AUTO scaling techniques
CPU fallback calls `context->Rescale` (a no-op for FLEXIBLE*/FIXEDAUTO); the GPU path performs a
real rescale. Decide the intended api semantics and align (excluded from `EvalArithmetic` for now).

### O4. Chebyshev range-transform prelude not aligned
For bounds other than (−1, 1), the GPU applies add-then-mult on the input; CPU builds
`T[0] = EvalMult(x, α); ModReduce; EvalAddInPlace(−1−β)`. Different rounding order. All current
tests use (−1, 1). Align the GPU prelude to the CPU order when needed.

### O5. Chebyshev degree < 5 takes `EvalChebyshevSeriesLinear` on CPU
The GPU has no mirrored linear path. Only matters if low-degree series are used through the api.

### O6. Untested configurations (Tier 2)
- Bootstrap with level budget {1,1} → `isLT`/`EvalLinearTransform` branch (untouched by all work so far).
- FLEXIBLEAUTOEXT (OpenFHE default) — extra-level ModRaise handling.
- FIXEDMANUAL branches of the Chebyshev/bootstrap transcriptions (mirrored by reading, never validated).
- Other slot counts (64, 512, …); `SPARSE_ENCAPSULATED` key distribution path.

### O7. api type-erasure design (`std::any`) — decide a direction
The api layer stores its OpenFHE/FIDESlib objects type-erased (`std::any cpu/gpu/pimpl`,
`shared_ptr<void>` GPU registry) so the public headers carry no OpenFHE or CUDA includes.
The types are fully known at compile time inside the `.cpp` files — the erasure exists only for
the header boundary, because OpenFHE's public types are backend-configured alias templates that
are unsafe to forward-declare. Costs: `any_cast` turns type errors into runtime throws (R10 is
the canonical failure mode), a per-access type check, and an extra heap allocation per `any`
holding a `shared_ptr` (exceeds libstdc++'s small-object buffer). The boundary is also already
punctured: `GetElements()`/`GetEncodingType()` return `lbcrypto` types, so `Ciphertext.hpp` now
includes two OpenFHE headers. Pick one direction:
- **A — restore header independence:** remove the OpenFHE includes from api headers again. Requires
  reworking `GetElements()` (opt-in `openfhe_interop.hpp` header, or an opaque/`std::any` return)
  and mirroring `PlaintextEncodings` in `Definitions.hpp` like the other mirrored enums. Keep
  `std::any` internally but centralize all casts in one internal helper header (typed accessors,
  each stored type spelled exactly once) so R10-class bugs cannot recur.
- **B — remove `std::any`:** store the concrete types (or a typed pimpl struct) — compile-time
  safety, no per-access checks, no extra allocation. api headers then include OpenFHE outright;
  consumers need OpenFHE headers on their include path (they already link against it).
Deciding factor: whether external consumers must compile against `fideslib.hpp` without an
OpenFHE installation. Either way, the centralized-cast hardening is worth doing immediately.

## Plan forward

**P1 — Land the current state.** I'll push a branch with the current changes (FIDESlib fixes,
`deps/fideslib-ref-1.5.1.2.patch` + `build.sh` bump, `test/OpenFheCompatTests.cu`, and this
document). The green suite is the regression wall for everything below.

**P2 — Upstream OpenFHE PRs** (shrinks the patch back to visibility shims; each step re-validated
with the stage harness):
- **a.** *(done)* `EvalRotate`/`EvalAtIndex`/`EvalAutomorphism` + `Conjugate` unification onto
  the hoisted core (O1) — landed as `fideslib-ref-v1.5.1.2`; the R1 interim hunks are dropped
  from the patch and `OpenFHECompatTests.EvalRotate` is green. Remaining: propose the same
  change to the upstream OpenFHE dev branch proper.
- **b.** Lazy extended-basis PartialSum (also a CPU win: same mod-downs saved). On merge: restore
  the lazy `Accumulate` in FIDESlib (reverts R2's perf cost). Subtlety: the initial lift of
  standard `c0` into the extended basis must match FIDESlib's convention — pin with one bisect cycle.
- **c.** Lazy Horner giant-step in `EvalHornerGiantRotate` (our own upstream code): carry `c0`
  extended through the Horner loop, mod-down `c1` only, settle at level end. On merge: flip
  `ONLY_C1` back to true (reverts R3's perf cost).
- If upstream declines b/c: either carry them in the patch (grows it) or keep FIDESlib eager
  (current state, already validated).

**P3 — Fix the `rotate_hoisted` memory bug (O2)** via ASAN host build or gdb watchpoint;
re-enable `DISABLED_EvalFastRotationHoisted`.

**P4 — Tier-2 coverage (O6)** one configuration at a time — each red result is a mini-investigation
with the harness. Then close out the smaller semantic gaps (O3–O5) and the api type-erasure
decision (O7).

## Validation methodology (reusable)

Env-gated bisection: add a `BOOT_STAGE_LIMIT` helper to both pipelines
(`ckksrns-fhe.cpp` / `Bootstrap.cu` + `ApproxModEval.cu`), truncating after aligned stages —
1 ModRaise, 2 PartialSum, 3 CtS, 4 conj+add, 5 approxMod (51x = Chebyshev sub-stages: baby powers,
giant powers, T2km1, inner PS; 52 = double-angle), 6 StC(+doubling); dense variants 71–82 —
then compare with the exact-equality test. One build+run per probe localizes any divergence to a
single stage. Gotchas learned: `g_used_rot_indices`-style key instrumentation misses
`GetRotationKey`'s alternative-key early-return (the "unused key" that started all this was an
artifact); `normalyzeIndex` maps rotation indices through `ctxt.slots`; encryption is randomized,
so CPU/GPU test phases must share a single `Encrypt` call.

## Performance accounting (to recover via P2)

| Change | Cost of current (eager) form |
|---|---|
| R2 `Accumulate` | ~7 extra `c0` `ApproxModDown`s per bootstrap, at raised level |
| R3 `ONLY_C1=false` | ~1 extra `c0` mod-down per giant step per CtS/StC level |
| R4 Chebyshev transcription | temp-ciphertext clones + k−1 alignment ops (CPU does these too); optimize buffer reuse only if profiling warrants |

Numbers are op-count based (slots=8, budget {3,3}); profile before prioritizing.
