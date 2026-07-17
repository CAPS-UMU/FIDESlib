# CPU/GPU Bit-Compatibility: Issues and Plan

> **Note:** this is a working document tracking the bit-compatibility campaign. It will be
> removed once everything is ready to merge.

**Goal:** every FIDESlib GPU operation produces ciphertexts that are bit-identical to the
patched-OpenFHE CPU equivalent, verified by exact ciphertext comparison (`ASSERT_EQ_CIPHERTEXT`),
not decrypt-level tolerance. Exact plaintext-level comparison is impossible by design: OpenFHE
adds fresh Gaussian noise to every CKKS decryption (`CKKSPackedEncoding::Decode` noise-flooding
countermeasure), so decrypted values are compared within precision (`ASSERT_ERROR_OK`) only.

**Landing zones.** Changes live in one of three places, in order of preference:

1. **OpenFHE dev branch** — algorithmic changes with standalone value (formulation unification,
   optimizations that also help the CPU).
2. **`deps/fideslib-ref-1.5.1.4.patch`** — kept minimal; visibility/build-config shims only
   (applies on tag `fideslib-ref-v1.5.1.4`). Temporary home for changes queued for upstream.
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
| `AccumulateSum` | PASS | api rotation fold: CPU fallback ≡ GPU `Accumulate(bStep=2)` (R11) |
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

**Resolution (O1, `fideslib-ref-v1.5.1.2`):** `EvalRotate`/`EvalAtIndex`/`Conjugate` are routed
through the hoisted core upstream, so the original bootstrap call sites are bit-compatible as
written — the five interim hunks were reverted to stock upstream code and dropped from the
patch (they were never upstreamed as call-site changes).

### R2. `Accumulate` lazy extended-basis accumulation *(deliberate GPU optimization — perf recovered via P2b)*
The GPU PartialSum kept `c0` in the Q·P basis across all doubling levels with one deferred
mod-down. `ApproxModDown` rounds, so `moddown(Σx) ≠ Σ moddown(x)`; OpenFHE's per-rotation
mod-down can never match.

**Interim fix (FIDESlib `AccumulateBroadcast.cu`):** sequential full rotations, per-rotation
mod-down. Perf cost ≈ 7 extra `c0` mod-downs per bootstrap at the raised level (the hoisting
itself was moot: bootstrap uses `accumulate_bStep = 2`, one rotation/level).

**Resolution (P2b, `fideslib-ref-v1.5.1.3`):** the CPU PartialSum is lazy too
(`FHECKKSRNS::EvalPartialSumInPlace`, see P2b), and the lazy `Accumulate` was restored in
FIDESlib.

### R3. `LinearTransform` `ONLY_C1` lazy mod-down *(deliberate GPU optimization — perf recovered via P2c)*
Horner giant-step and correction rotations mod-downed only `c1`, keeping `c0` extended; CPU
`EvalHornerGiantRotate` does a full `KeySwitchDown` (with exact `·PModq` re-lift) per giant step.

**Interim fix (FIDESlib `LinearTransform.cu`):** `ONLY_C1 = false`. Perf cost ≈ 1 extra `c0`
mod-down per giant step per CtS/StC level.

**Resolution (P2c, `fideslib-ref-v1.5.1.3`):** `EvalHornerGiantRotate` settles only `c1` and
folds the extended `c0` directly into the key-switch product; the CtS/StC slot-ordering
corrections fold into the last level's extended output; `ONLY_C1` is back to `true` (see P2c).
The level-end `KeySwitchDown` is unchanged and settles `c0` once per level. FIDESlib's extended
representation is ·P-scaled exactly like the key-switch product, so the direct fold is
bit-compatible by construction (validated by the suite, no bisect cycle needed).

### R4. GPU Chebyshev was a stale port (values wrong, not just slower)
`evalChebyshevSeries`/inner PS ported an older OpenFHE: mutated shared powers `T[i]` in place
(sticky adjustments), skipped the `AdjustLevelsAndDepthInPlace` power alignment, and the inner
recursion didn't match the current qu/su/cu construction.

**Resolution (FIDESlib `ApproxModEval.cu`):** line-for-line transcription of current OpenFHE
(clone-based operand adjustment, power alignment, transcribed inner PS +
`EvalPartialLinearWSum`). Locked in by `EvalChebyshev` + both bootstrap tests.

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
silently cost callers a level.

**Resolution (`api/CryptoContext.cpp`):** exact per-limb multiply by `q_i − 1`; no metadata
change. Applies to `EvalNegate` and `EvalNegateInPlace`.

### R10. api `bad any_cast` in ct×pt CPU fallbacks
`EvalMult(ct, pt)` / `EvalMultInPlace(ct, pt)` cast `pt->cpu` to `ConstPlaintext` where it holds
`Plaintext` — an `std::any` cast must name the stored type exactly, so any CPU-path ct×pt
multiply threw.

**Resolution (`api/CryptoContext.cpp`, both call sites):** cast to the stored type:

```diff
-		auto& ptImpl  = std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
+		auto& ptImpl  = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
```

This failure mode is inherent to the `std::any` design — see O7.

### R11. api `AccumulateSum` CPU fallback mismatched the GPU accumulation
The GPU path runs `FIDESlib::CKKS::Accumulate(ct, bStep=4, stride, slots)` — radix-4 levels,
each sharing one digit decomposition across up to 3 rotations, with the lazy extended-basis
`c0` accumulation — while the CPU fallback was an eager sequential doubling loop
(`EvalRotate` + `EvalAddInPlace`): a structurally different accumulation order, so the two
paths never matched bitwise. Additionally, the GPU path narrows the FIDESlib `slots` field to
`stride` after a full fold (its broadcast convention), which leaked through api `GetSlots` and
made the GPU result decode at a different length than the CPU result — a pre-existing
api-visible inconsistency.

**Resolution (`fideslib-ref-v1.5.1.4`):** `FHECKKSRNS::EvalPartialSumInPlace` provides a static
`(ct, stride, size)` radix-2 helper (doubling; one rotation per level, folded directly into the
extended accumulator) plus a general `(ct, stride, size, radix)` form that delegates to the
radix-2 helper when `radix == 2`; both mirror FIDESlib's `Accumulate` level/index structure
bit-for-bit. A single api constant `ACCUMULATE_SUM_RADIX = 2` drives both sides: the CPU
fallbacks call the general form with it, and the GPU calls `Accumulate(*, ACCUMULATE_SUM_RADIX,
…)` (FIDESlib's existing `bStep` parameter — no new GPU code). The api GPU paths also restore
the OpenFHE-visible slot count after the fold. Radix-2 keeps the rotation-key footprint at the
power-of-two set (see O8, radix/key-size tradeoff — key minimization drove this choice).
Acceptance test: `OpenFHECompatTests.AccumulateSum`. The `start`-offset variant
(`AccumulateCascadeImpl`, which settles both elements per level) still uses `bStep=4` on the GPU
and keeps its eager doubling fallback — deferred (see O8).

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

### O8. Rotation-fold radix — performance vs key-size scaling knob
Every rotation-accumulation fold (`sum_j Rotate(ct, j·stride)` over `size` summands) can be
evaluated at any power-of-two **radix** — the accumulation branching factor, exposed as the
`radix` parameter of `FHECKKSRNS::EvalPartialSumInPlace(ct, stride, size, radix)` (OpenFHE) and
the `bStep` parameter of FIDESlib's `Accumulate(ct, bStep, stride, size)` (GPU; `bStep` is a
BSGS-carryover name — for a pure rotation fold it is the radix, not a baby-step count). The
radix trades digit decompositions against rotation keys:

- **Digit decompositions (mod-ups, the dominant cost):** `log_radix(size)` levels, one
  decomposition each → `log2(size) / log2(radix)` mod-ups. Radix 4 does half as many as
  doubling; radix 8, a third.
- **Rotation keys:** `(radix − 1)` distinct rotations per level →
  `(radix − 1)·log_radix(size) = (radix − 1)/log2(radix) · log2(size)` keys. Doubling = `log2(size)`;
  radix-4 = `1.5·log2(size)`; radix-8 ≈ `2.33·log2(size)`.

The catch that makes doubling especially cheap on keys: **radix-2's index set is exactly the
power-of-two rotations `{stride·2^i}`** — the universal set essentially every CKKS program
already generates, so its *incremental* key cost is ≈ 0. Higher radices add *dedicated*
non-power-of-two indexes unlikely to be shared: radix-4 adds `{3·stride·4^j}`
(`{3, 12, 48, 192, …}` for stride=1), i.e. `floor(log4(size))` extra dedicated keys —
+1 at 8 slots, +5 at 1024, +7 at fully-packed 32768. At bootstrapping parameters a single
rotation key is tens of MB, so those extra keys are real storage.

**Current choices (key minimization is a priority — see the memory note):** everything uses
radix-2. Bootstrap PartialSum runs `accumulate_bStep = 2` (FIDESlib's name for the radix); api
`AccumulateSum` uses radix-2 on both CPU and GPU (R11). The general `(…, radix)` helper is
retained but unused otherwise, so a future
mod-up-bound workload that can afford the keys can opt a specific fold into a higher radix
without new machinery. **Do not raise any radix without surfacing the key delta and getting a
decision.** Open sub-item: the `start`-offset `AccumulateSum` variant (`AccumulateCascadeImpl`)
still runs `bStep=4` on the GPU with an unvalidated eager-doubling CPU fallback; bringing it to
radix-2 (and under the compat test) would close the last AccumulateSum key gap.

## Plan forward

**P1 — Land the current state.** I'll push a branch with the current changes (FIDESlib fixes,
`deps/fideslib-ref-1.5.1.4.patch` + `build.sh` bump, `test/OpenFheCompatTests.cu`, and this
document). The green suite is the regression wall for everything below.

**P2 — Upstream OpenFHE PRs** (shrinks the patch back to visibility shims; each step re-validated
with the stage harness):
- **a.** *(done)* `EvalRotate`/`EvalAtIndex`/`EvalAutomorphism` + `Conjugate` unification onto
  the hoisted core (O1) — landed as `fideslib-ref-v1.5.1.2`; the R1 interim hunks are dropped
  from the patch and `OpenFHECompatTests.EvalRotate` is green. Remaining: propose the same
  change to the upstream OpenFHE dev branch proper.
- **b.** *(done)* Lazy extended-basis PartialSum, implemented as
  `FHECKKSRNS::EvalPartialSumInPlace` and used at all three PartialSum sites (`EvalBootstrap`,
  `EvalBootstrapStCFirst`, `EvalHomDecoding`): element 0 accumulates in the extended (QlP)
  basis starting from `P·cv[0]` with a single deferred `ApproxModDown`; element 1 settles per
  level (its digit decomposition feeds the next rotation). Also a CPU win: `log2(N/(2·slots))−1`
  fewer mod-downs per PartialSum (7 for slots=8, N=4096). The initial-lift convention matched
  FIDESlib directly — FIDESlib's mixed standard/extended adds P-scale the standard operand
  (`add_scale_p_*` kernels), identical to OpenFHE's `·PModq` lift, and
  `ApproxModDown(P·x + ks) = x + ApproxModDown(ks)` holds exactly, so no bisect cycle was
  needed. The lazy `Accumulate` is restored in FIDESlib (R2 perf recovered); full suite green.
  Landed with P2c as `fideslib-ref-v1.5.1.3` (commit `d31322ac`); remaining: propose upstream.
- **c.** *(done)* Lazy Horner giant-step: `EvalHornerGiantRotate` settles only
  `c1` (its digit decomposition feeds the key switch) and folds the extended `c0` into the
  key-switch product directly — no per-giant-step `ApproxModDown` + `·PModq` re-lift round-trip.
  Additionally, the final CtS/StC slot-ordering corrections (`EvalAtIndex(result, Delta)`) are
  folded into the last level's Horner output while it is still extended, matching the GPU's
  offset rotation (`LinearTransform`'s `offset` consumes the extended `c0` under `ONLY_C1`).
  `EvalLinearTransform` inherits the lazy giant steps automatically. FIDESlib `ONLY_C1` is back
  to `true` (R3 perf recovered); full suite green, dense bootstrap ~11% faster. The same lazy
  folds were applied to the scheme-switching linear transforms (`EvalLTWithPrecomputeSwitch`
  gains an extended-output flag; the sparse SlotsToCoeffs doubling and the wide-matrix log-fold
  in `EvalLTRectWithPrecomputeSwitch` accumulate extended and settle once) — CPU-only paths, no
  GPU counterpart, validated with OpenFHE's pke unit tests. Landed with P2b as
  `fideslib-ref-v1.5.1.3` (commit `d31322ac`); remaining: propose upstream.
- **Considered, not pursued — radix-4 bootstrap PartialSum (`accumulate_bStep = 4`):** the
  generalized `EvalPartialSumInPlace(ct, stride, size, radix)` (R11) supports raising the
  PartialSum radix on both sides, which would halve the digit decompositions at the raised
  level (8 → 4 for slots=8, N=4096) — the widest-tower, most expensive mod-ups of the
  bootstrap. The cost is ~1.5× more rotation keys for the PartialSum indexes (radix 4 needs
  keys for `{1,2,3}·stride·4^i` vs doubling's `stride·2^i`). **Minimizing key storage is a
  priority, so we stay at `accumulate_bStep = 2`.** Revisit only if the raised-level mod-up
  cost ever dominates profiles badly enough to outweigh the key growth.

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
| R2 `Accumulate` | *recovered (P2b)* — lazy accumulation restored on both sides |
| R3 `ONLY_C1=false` | *recovered (P2c)* — lazy Horner restored on both sides |
| R4 Chebyshev transcription | temp-ciphertext clones + k−1 alignment ops (CPU does these too); optimize buffer reuse only if profiling warrants |

Numbers are op-count based (slots=8, budget {3,3}); profile before prioritizing.
