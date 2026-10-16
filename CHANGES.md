# Changelog

All notable changes to FIDESlib are documented in this file. This release is the
result of the `cleanup` integration branch (merged PRs #47, #48 and #49) and the
work it accumulated since the previous release, **FIDESlib 2.1.3**. It closes
issues #11, #15, #29, #32, #33 and #42, and supersedes pull requests #18, #45
and #46.

## FIDESlib 2.2.0

FIDESlib 2.2.0 is the **CPU/GPU bit-compatibility** release. Every GPU operation
that OpenFHE can evaluate on the CPU now produces a ciphertext that is *bit-identical*
to the patched-OpenFHE CPU reference, verified with exact ciphertext comparison
(`ASSERT_EQ_CIPHERTEXT`) rather than decrypt-level tolerance. That includes the
previously value-only cases: SPARSE_ENCAPSULATED bootstrapping and the start-offset
rotation fold. The release also adds SlotsToCoeffs-first and iterative bootstrapping,
complex-valued CKKS, the server→client return path, a full local coverage/tooling
pipeline, and portable NTT/INTT kernels.

### Requirements

- **OpenFHE 1.6.1 (patched)** is now required (`find_package(OpenFHE 1.6.1 ... EXACT)`),
  replacing the previous `fideslib-ref-v1.5.1.1` reference.
- The auto-installer (`FIDESLIB_INSTALL_OPENFHE=ON`) gains selectable source modes and
  now builds OpenFHE's **static** libraries so FIDESlib can link against the installed
  prefix:
  - `FIDESLIB_OPENFHE_SOURCE=patch` (default) — official v1.6.1 + `deps/fideslib-1.6.1-compile.patch`,
    the full set of reference changes FIDESlib matches bit-for-bit.
  - `FIDESLIB_OPENFHE_SOURCE=patch-minimal` — v1.6.1 + `deps/fideslib-1.6.1-compile-minimal.patch`,
    only the compilation/API surface FIDESlib builds against (no bit-exact reference changes).
  - `FIDESLIB_OPENFHE_SOURCE=port` — the fully modified 1.6.1-fideslib fork branch.
- **Non-FMA codegen guard.** FIDESlib reproduces OpenFHE's Chebyshev / functional-bootstrapping
  arithmetic bit-for-bit, which requires both sides to use identical (non-FMA) floating-point
  codegen. CMake now warns at configure time if OpenFHE was built with FMA-enabling flags, and
  a runtime probe (`OpenFheCompat`, `FIDESLIB_OPENFHE_FMA_FLAGS`) throws unless
  `FIDESLIB_ALLOW_OPENFHE_FMA=1` is set. Build OpenFHE with `-DWITH_NATIVEOPT=OFF`.

### Bootstrapping

- **SlotsToCoeffs-first (decode-first) bootstrap** — `EvalBootstrapStCFirst(ct, numIterations, precision)`
  and the `btsfirstboot` flag on `EvalBootstrapSetup`. Includes the FLEXIBLEAUTO/FLEXIBLEAUTOEXT
  depletion correction, the sparse-decode slot fix (sparse case doubles the decoded message under
  the REAL default), and the FIXEDAUTO tower drop.
- **Iterative bootstrap** — `EvalBootstrap`/`EvalBootstrapInPlace` accept `numIterations`
  (second error-correcting pass) and `precision`, with the level alignment of the first `EvalSub`
  and the FIXEDAUTO-only tower drop handled correctly. All four scaling techniques
  (FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO, FIXEDMANUAL) are covered.
- **Per-slot bootstrap-mode guardrails** — a new `BootstrapMode` (UNCONFIGURED / STANDARD /
  STC_FIRST / FBT / FEFBT) prevents evaluating one bootstrap mode against a precomputation
  configured for the other (which would silently misroute). `isComplex` is populated from the
  OpenFHE CKKS datatype.
- **Latest Chebyshev / key-switch stack** — T2_norescale GPU reorder, fused complex weighted sum
  (WSum), `bootK = 1.0` `GetRawParams` fix (`g_coefficientsSparse` / SPARSE_TERNARY), the
  non-FMA `ElemForEvalMult`, and compressed 1-slot constant plaintexts that keep only the used
  slots (sparse/stride-compressed operands).

### API

- **Server→client return path** — `CryptoContextImpl::UnloadCiphertext(ct)` materializes the host
  copy (including the extended/modUp basis) and evicts the GPU ciphertext; `SerializeToFile` /
  `DeserializeFromFile` overloads for `Ciphertext` round-trip it. A GPU server can now compute,
  unload, serialize and ship results without holding the secret.
- **Mutable operations** — every `Eval*Mutable*` entry point is wired to the FIDESlib `*Mutable`
  GPU primitives and OpenFHE's mutable ops, mutating operands in place and matching the immutable
  forms bit-for-bit.
- **Complex-valued CKKS** — the public `CCParams<CryptoContextCKKSRNS>::SetCKKSDataType(COMPLEX)`
  / `GetCKKSDataType()` API is exposed (new `CKKSDataType` enum in `api/Definitions.hpp`), so
  `MakeCKKSPackedPlaintext` carries the imaginary parts through the encoding and complex ciphertext
  arithmetic runs on the GPU. The internal complex machinery (complex scalar `addScalar`/`multScalar`,
  the fused complex weighted sum) already backed complex bootstrapping; the missing piece was the
  public datatype switch. Covered by `OpenFHECompatTests.ComplexDataType` (encoding + GPU
  arithmetic) and `OpenFHECompatTests.ComplexBootstrap` (CPU and GPU complex bootstrap agree).
- **Sparse plaintext compression** — a plaintext created with fewer than the maximum slots
  (`MakeCKKSPackedPlaintext(..., slots)`) is stored **compressed** on the GPU: `RNSPoly::loadConstant`
  keeps only the used slots per limb (a stride-decimated `2*slots`-value representation) instead of
  `N` values, and the arithmetic kernels consume that layout directly through their
  strided/broadcast operand paths. This cuts VRAM for the bootstrap precomputation matrices
  (CtS / StC / LT) and for large constant plaintexts.
- `SetLevel` is now a **pure tower drop** on the GPU (matching OpenFHE, which never touches values).
- `EvalNegate` is an exact negation with no extra depth; the plaintext/ciphertext CPU fallbacks'
  bad `any_cast` is fixed.
- Guardrails converted from silent `assert` / `exit(-1)` / empty-return paths to exceptions
  (`GetBootPrecomputation`, `GetSecretSwitchingKey`, the `RawCiphertext` missing-key lookups, and
  the remaining `Ciphertext` adjust-failure paths, which previously default-constructed in release
  builds).

### Performance

- **Lazy Shoup / lazy modular reduction** across the arithmetic kernels.
- **Portable 1D NTT/INTT kernels** (`src/NTT_1D.cu`, `src/INTT.cu`, `src/INTT_1D.cu`) and a
  reworked NTT stack; the fusion kernels adopt the bit-identical **one-multiply rescale**
  (`(in - A) * q_l^-1`), removing the dead all-pairs `QlQlInvModqlDivqlModq` table (host mirror +
  device global) and one multiply + one global load per element.
- **Streamed weighted sum** in `evalPartialLinearWSumCompat` (one scaled term live at a time),
  dropping peak live temporaries from `limit + 1` ciphertexts to 3 at the widest bootstrap levels —
  important on small cards.
- Chebyshev / Paterson–Stockmeyer evaluation optimizations.
- `sm_75` added to the default CUDA architecture list.

### Bit-compatibility and interoperability

- New exhaustive `test/OpenFheCompatTests.cu` pins bit-exact CPU===GPU output for the whole
  ciphertext-producing API surface: arithmetic (ct∘ct, ct∘pt, ct∘scalar, in-place, mutable, swapped
  argument order), rotations (`EvalRotate`, `EvalFastRotation`, hoisted multi-index, extended-basis
  multi-index), Chebyshev, accumulate, all bootstrap modes and their FIXEDMANUAL/FIXEDAUTO/
  FLEXIBLEAUTOEXT combinations, `SetLevel`, extended-basis export, and the server→client path.
- **Extended (modUp) basis round-trip** — `Ciphertext::store`/`RNSPoly::store` and the api download
  path include the special/`P` towers, so `EvalFastRotationExt` (single- and multi-index) produces
  the same 5-tower `QL·P` result as the CPU reference.
- **FIXEDAUTO** plaintext degree raise in `Plaintext::adjustScaleAndLevel` mirrors the patched
  `AdjustLevelsAndDepthInPlace` (fixes `CiphertextAdjustMatrix` and the `ExtractContextShowPtMult*`
  cases).
- **FIXEDMANUAL** double-angle rescale placement and T2km1 level alignment fixed.
- **SPARSE_ENCAPSULATED bootstrap is bit-exact** against the in-context stock reference across
  the sparse ({3,3}), fully-packed, LT ({1,1}) and iterative (`numIterations=2`) configurations —
  the previously value-only O6c carve-out is resolved (`EvalBootstrapSparseEncaps`,
  `EvalBootstrapSparseEncapsDense`, `EvalBootstrapSparseEncapsLT`,
  `EvalBootstrapSparseEncapsIterative`).
- `Accumulate` unified into a single lazy radix fold driven by `ACCUMULATE_SUM_RADIX`
  (the previous radix-4 eager cascade is gone). The `start`-offset `AccumulateSum` variant is now
  **bit-exact** too: the patched OpenFHE reference's start-offset
  `EvalPartialSumInPlace(ct, stride, size, radix, startFactor)` runs the same lazy fold on the CPU
  (folded into `deps/fideslib-1.6.1-compile.patch` and the fork branch `1.6.1-fideslib`).
- `src/CKKS/openfhe-interface` ported to the mainline 1.6.1 API (canonical sparse-parameter getters,
  the 7-argument `EvalBootstrapSetup` with `modevallevels`).

### Tooling, testing and diagnostics

- **Local coverage pipeline** — `scripts/run_coverage.sh` and the sharded, resumable
  `scripts/run_coverage_sharded.sh` (one test case per process, crash-isolated, accumulating gcov
  counters, resumable via a done-file). Baseline figures and caveats are recorded in
  [COVERAGE.md](COVERAGE.md): api line coverage **96.7 %** / functions **100 %**, api-core view
  95.7 % lines / 99.5 % functions on the full suite.
- **Static kernel-usage report** — `scripts/kernel_usage.py` + [KERNEL_USAGE.md](KERNEL_USAGE.md)
  classify every `__global__` kernel under `src/` and `api/` by reachability and cross-check launch
  sites against the gcov report (gcov cannot instrument device code).
- **[BITCOMPAT.md](BITCOMPAT.md)** documents the CPU/GPU bit-compatibility campaign, its landing
  zones and the resolved issues (R1–R15).
- **NVTX memory-aware lifetimes** — object timelines render as `N x Label - X.XX MB`; the memory
  pool gets its own `FIDESlibPool` row (`Chunks of 512KB (2048 chunks) = 1GB`); the `Context` row
  reports its aggregate footprint with a `Precomputed`/`Buffers` breakdown, and re-renders are
  de-duplicated.
- Coverage builds skip `nvcc -G` (gcov only needs host data), fixing `too many resources requested
  for launch` on large kernels.

### Bug fixes

- Issue #42: `evalChebyshevSeries` mapped the interval into `[-1, 1]` with half the interval
  **width** instead of the **midpoint**; every interval not symmetric about 0 (or starting at 0)
  evaluated the polynomial at the wrong point. Both branches now shift by the midpoint, matching
  OpenFHE.
- Issue #15: the NTT/OpenFHE test failures reported against the 2.0 release are resolved — the NTT
  suites are rewritten against the current kernels and the OpenFHE interface/bootstrap suites are
  bit-exact.
- Issue #31: asymmetric Chebyshev-interval bootstrap failure.
- `rotate_hoisted` non-fused path copies all metadata to the results; the `ext` contract is documented.
- `LimbPartition::multPt` uses the top limb of the current level (FIXEDMANUAL `multPt` with rescale
  was wrong below the top level).
- `addPt`/`subPt`/`multPt` reject a plaintext with too few limbs instead of silently producing a
  wrong result.
- `Accumulate`/`Broadcast` reject `bStep` values that would hang or give wrong sums.
- Bootstrap rejects inputs with fewer limbs than their `NoiseLevel` needs (previously returned garbage).
- The CUDA stream pool is created lazily instead of from a static initializer.
- `clearAuxilarPoly` synchronizes once instead of recording an event per pooled buffer.
- `CiphertextImpl` copy-constructor infinite recursion fixed (a `shared_ptr` was `static_cast` to
  the reference type through the very constructor being defined).
- `Limb` size asserts corrected for the `num_elem = -1` default (Debug-only failure).
- `ASSERT_EQ_CIPHERTEXT` now evaluates its arguments exactly once; the suspected `rotate_hoisted`
  heap corruption (BITCOMPAT O2/P3) was a test-harness bug, and `EvalFastRotationHoisted` is
  re-enabled.

### Removed / changed

- `test/BugTests.cu` removed (superseded by the compat/interface suites).
- `deps/openfhe-1.5.1.patch` and `deps/fideslib-ref-1.5.1.1.patch` removed; superseded by the
  OpenFHE 1.6.1 patches.
- `deps/draft-start-accumulate-radix-lazy.patch` removed; folded into
  `deps/fideslib-1.6.1-compile.patch` (and the OpenFHE fork branch `1.6.1-fideslib`).
- The dead `QlQlInvModqlDivqlModq` table (host mirror and device global) removed.
- `clang-format` applied across the tree.

### Fixed issues

| Issue | Title | Resolution |
|---|---|---|
| **#42** | `evalChebyshevSeries` maps the domain with half the interval WIDTH instead of the MIDPOINT | Fixed: both affine-map branches shift by the midpoint, matching OpenFHE. |
| **#33** | CKKS bootstrapping incorrect under SPARSE_ENCAPSULATED on GPU (garbage output; SIGSEGV with `numIterations=2`) | Fixed: SPARSE_ENCAPSULATED bootstrap is **bit-exact** across sparse/dense/LT/iterative; the `numIterations=2` SIGSEGV is gone. |
| **#32** | Plans to add StC-first bootstrapping? | Implemented: `EvalBootstrapStCFirst` + the `btsfirstboot` setup flag (all scaling techniques). |
| **#29** | Iterative bootstrapping | Implemented: `EvalBootstrap(..., numIterations, precision)` (all four scaling techniques). |
| **#15** | Multiple NTT and OpenFHE Test Failures (FIDESLIB Release 2.0) | Fixed: NTT suites rewritten against the current kernels; OpenFHE interface/bootstrap suites are bit-exact. |
| **#11** | Complex-values Bootstrapping | Implemented: public `SetCKKSDataType(COMPLEX)` / `GetCKKSDataType()`; complex encoding, arithmetic and bootstrap covered by tests. |

The Chebyshev-interval family from the same report is also addressed: **#31** (asymmetric
Chebyshev-interval bootstrap failure) and **#24** (interior coefficients exactly `0.0`), both
previously closed and carried by this line.

### Superseded pull requests

| PR | Title | Why it is redundant |
|---|---|---|
| **#45** | Fix: GPU compatibility for V100, with CUDA 12.x and sm_70 | The `nvidia-smi` architecture auto-detection + portable default (incl. `70-real`) and the associated `MAXP`/level/limb-count/API fixes are all present in this tree. |
| **#18** | Added complex scalar addition and complex scalar multiplication | Implemented here: complex `addScalar`/`multScalar`, the fused complex WSum, the complex CKKS datatype API, and complex bootstrapping (see #11). |
| **#46** | Version 2.1.4 | Closed earlier; this release supersedes it (2.2.0). |

The merged PRs #47 (NVTX memory lifetimes), #48 (coverage/tooling) and #49 (StC + iterative
bootstrap + OpenFHE 1.6.1) are part of this release.

### Known issues

- **Multi-GPU** (`LimbPartitionMGPU.cu`) is not exercised by the single-GPU test/coverage run
  (see KERNEL_USAGE.md). Known multi-GPU limitations (peer-copy transport and P2P probing on
  hosts where `cudaDeviceCanAccessPeer` cannot be trusted, `ASSERT_ERROR_OK` NaN detection) are
  tracked as future work (issue #28) and are unrelated to the single-GPU bit-compatibility of
  this release.
- The batched (`LimbPartitionBatch.cu`) kernel paths are not exercised by the single-GPU
  coverage run (see KERNEL_USAGE.md).
- The only remaining note is a **non-compatibility** design decision, tracked in
  [BITCOMPAT.md](BITCOMPAT.md) (O7): the api type-erasure direction (`std::any` at the public
  header boundary). All bit-compatibility items (O1–O6, O8–O11) are resolved.

### Verification

Full test suite against the **full patched OpenFHE 1.6.1** reference
(`deps/fideslib-1.6.1-compile.patch`, `FIDESLIB_OPENFHE_SOURCE=patch`), `FIDESLIB_ARCH=89-real`,
on an RTX 4060 Ti (8 GB):

| Suite | Result |
|---|---|
| `OpenFHECompatTests` | 75 / 75 pass |
| `OpenFHEBootstrapTests` | 88 / 88 pass |
| `OpenFHEInterfaceTests` | 192 / 192 pass |
| `NTTTests` | 48 / 48 pass |
| `NegacyclicNTTTests` | 20 / 20 pass |
| `MathTests` | 5 / 5 pass |
| `AccumulateBroadcast` | 2 / 2 pass |
| `BaseTests` | 25 / 25 pass |
| `LimbKernelTests` (incl. `LimbKernelTest32`) | 26 / 26 pass |
| `OpenFHECompatTests.ComplexDataType` (new) | pass |
| `OpenFHECompatTests.ComplexBootstrap` (new) | pass |
| `OpenFHECompatTests.EvalBootstrapSparseEncaps` (newly enabled, bit-exact) | pass |
| `OpenFHECompatTests.EvalBootstrapSparseEncapsDense` | pass (bit-exact) |
| `OpenFHECompatTests.EvalBootstrapSparseEncapsLT` | pass (bit-exact) |
| `OpenFHECompatTests.EvalBootstrapSparseEncapsIterative` (new) | pass (bit-exact) |
| `OpenFHECompatTests.AccumulateSumInPlaceStart` (newly enabled, bit-exact) | pass |
| `P2PTransferTest` | 2 pass, 3 skipped (single GPU) |
| `APIbench` / `P2PBenchmark` | pass (1 skip) |

The `LLMTests` timing suite is excluded: it provisions ~7 GB of rotation keys plus ~1.3 GB of
plaintexts at 32768 slots and OOMs on an 8 GB card.
