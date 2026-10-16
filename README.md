<p align="center">
  <img src="https://github.com/CAPS-UMU/FIDESlib/blob/main/doxygen/FidesLogo.drawio.svg?raw=true" width="200">
</p>

# FIDESlib 2.2.0

A server-side CKKS GPU library fully interoperable with OpenFHE.

## Improvements in version 2.2.0

- **CPU/GPU bit-compatibility**: every GPU operation is bit-identical to the patched-OpenFHE
  CPU reference (verified with exact ciphertext comparison).
- **SlotsToCoeffs-first and iterative bootstrapping** (`EvalBootstrapStCFirst`, `numIterations`).
- **Complex-valued CKKS** encoding via `SetCKKSDataType(COMPLEX)`.
- **Server→client return path**: `UnloadCiphertext` plus ciphertext serialization.
- **OpenFHE 1.6.1 compatibility** with selectable patched-install modes and a non-FMA codegen guard.
- Portable 1D NTT/INTT kernels, lazy Shoup reduction and other performance work.
- A local coverage pipeline and a static CUDA kernel-usage report.
- Closes issues **#11, #15, #29, #32, #33, #42**; supersedes pull requests **#18, #45**.

See [CHANGES.md](CHANGES.md) for the full changelog (added features, fixed issues and
superseded pull requests).

## Features

- Full CKKS implementation: Add, AddPt, AddScalar, Mult, MultPt, MultScalar, Square, Rotate, RotateHoisted, Bootstrap.
- OpenFHE interoperability for FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO and FLEXIBLEAUTOEXT, with bit-exact CPU/GPU results.
- Dense, sparse and SlotsToCoeffs-first bootstrapping, optionally iterative.
- Real and complex CKKS slot encoding.
- Hardware acceleration with NVIDIA CUDA.
- High-performance NTT/INTT implementation.
- Hybrid Key-Switching.
- Multi-GPU support with NCCL.
- Sparse Secret Encapsulation support (bit-exact bootstrap).
- Server→client ciphertext unload and serialization.
- Many performance optimizations.

## Supported parameters

FIDESlib implements the CKKS-RNS scheme and interoperates **bit-exactly** with the patched
OpenFHE 1.6.1 for the following choices:

- **Scaling techniques**: `FIXEDMANUAL`, `FIXEDAUTO`, `FLEXIBLEAUTO`, `FLEXIBLEAUTOEXT`.
- **Secret-key distributions**: `UNIFORM_TERNARY`, `SPARSE_TERNARY`, `SPARSE_ENCAPSULATED`.
- **CKKS data types**: `REAL`, `COMPLEX` (`CCParams::SetCKKSDataType`).
- **Key-switching technique**: `HYBRID`, with configurable digit size and number of large digits.
- **Bootstrapping**: ModRaise-first (standard), SlotsToCoeffs-first (`EvalBootstrapStCFirst`) and
  iterative (`numIterations`); dense and sparse packing; level budgets such as `{5,4}`, `{3,3}` or
  `{1,1}`; FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO and FIXEDMANUAL.
- **Operations** (immutable, in-place and `*Mutable` forms where applicable): Add, Sub, Negate,
  AddScalar / AddPt, Mult, MultScalar / MultPt, Square, Rotate, RotateHoisted, FastRotation and its
  extended-basis variant, Chebyshev series, AddMany, Accumulate / Broadcast, Rescale, SetLevel,
  Bootstrap.
- **Key / context serialization**: eval-mult and automorphism keys, public/private/context
  round-trips, and ciphertext unload + (de)serialization for the server→client path.
- **GPU targets**: CUDA 12/13, `sm_70`–`sm_120`. The default build is portable across the
  architectures the installed `nvcc` supports; `FIDESLIB_ARCH=auto` builds for the visible GPUs.
- **Multi-GPU** via NCCL; single-GPU is fully supported and bit-exact.

## Citation

If you use FIDESlib on your research, please cite our ISPASS paper.

```bibtex
@inproceedings{FIDESlib,
  author    = {Carlos Agull{\'o}-Domingo and {\'O}scar Vera-L{\'o}pez and Seyda Guzelhan and Lohit Daksha and Aymane El Jerari and Kaustubh Shivdikar and Rashmi Agrawal and David Kaeli and Ajay Joshi and Jos{\'e} L. Abell{\'a}n},
  title     = {{FIDESlib: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs}},
  booktitle = {2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year      = {2025},
  note      = {Poster paper},
  url       = {https://github.com/CAPS-UMU/FIDESlib},
  publisher = {IEEE},
  address   = {Ghent, Belgium}
}
```

## Compilation

> [!IMPORTANT]
> Requirements:
>  - NVIDIA CUDA version 12 or 13 (the default build covers `sm_70`–`sm_120`).
>  - A **patched OpenFHE 1.6.1** (installed automatically — see below).
>  - GCC version >= 11.
>  - OpenMP development library.
>  - CMake version 3.25.2 or greater.
>  - (Optional) NVIDIA Collective Communications Library to enable Multi-GPU support.

### Requirements installation

For CUDA software stack, follow the official installation guides. For the remaining
dependencies, install them using the package manager or method of your choice.

On Ubuntu:

```bash 
apt install make build-essential cmake git
```

> [!NOTE]
> CMake package on Ubuntu may be older than expected; install it using snap, pip or build from source.

### FIDESlib compilation

In order to be able to compile the project, one must follow these steps:

- Clone this repository.
- Generate the Makefile with CMake.
- Build the project.

```bash
# First build: installs the patched OpenFHE 1.6.1, then builds FIDESlib.
cmake -S . -B build -DFIDESLIB_INSTALL_OPENFHE=ON
cmake --build build -j

# Subsequent builds (the patched OpenFHE is already installed):
cmake -S . -B build -DFIDESLIB_INSTALL_OPENFHE=OFF
cmake --build build -j
```

`FIDESLIB_ARCH` selects the target architectures, e.g. `-DFIDESLIB_ARCH=89-real` for a fast
Ada-only build, or `-DFIDESLIB_ARCH=auto` to build only for the GPUs visible via `nvidia-smi`.

FIDESlib needs a **patched OpenFHE 1.6.1** in order to access some internals needed for
bit-exact interoperability. This patched version can be installed automatically by setting
`FIDESLIB_INSTALL_OPENFHE=ON` (default `OFF`); once installed, the flag can be turned off on
subsequent FIDESlib builds. The patched install builds OpenFHE's **static** libraries, which
FIDESlib links against.

Three ways of obtaining the reference are available (via `FIDESLIB_OPENFHE_SOURCE`):

| Mode                       | Source                                                                 | Use when |
|----------------------------|------------------------------------------------------------------------|----------|
| `patch` *(default)*        | Official OpenFHE `v1.6.1` + `deps/fideslib-1.6.1-compile.patch`         | Normal builds; the full bit-exact reference. |
| `patch-minimal`            | Official OpenFHE `v1.6.1` + `deps/fideslib-1.6.1-compile-minimal.patch` | You only need the compile/API surface (drops the bit-exact reference changes). |
| `port`                     | The fully modified `1.6.1-fideslib` fork branch                         | You want the upstream-ready fork instead of the patch. |

The installer is `deps/build.sh <install-prefix>` and can also be run directly. If you already
have a patched OpenFHE 1.6.1, point `OPENFHE_INSTALL_PREFIX` at it and set
`FIDESLIB_INSTALL_OPENFHE=OFF`.

> [!WARNING]
> FIDESlib reproduces OpenFHE's Chebyshev / functional-bootstrapping arithmetic bit-for-bit,
> which requires identical (**non-FMA**) floating-point codegen on both sides. CMake warns if
> the found OpenFHE was built with FMA-enabling flags, and a runtime probe throws unless
> `FIDESLIB_ALLOW_OPENFHE_FMA=1` is set. Build OpenFHE with `-DWITH_NATIVEOPT=OFF`.

The build process produces the following artifacts:

- fideslib.so: The FIDESlib library to be dynamically linked to any client application.
- fideslib-test: The test suite executable if selected.
- fideslib-bench: The benchmark suite executable if selected.
- gpu-test: A dummy executable to search for the CUDA capable devices on the machine.
- dummy: Another dummy executable.

The following options can be used with CMake to configure the build. The default value for each option is denoted in
**boldface** under the **Values** column:

| CMake Option                  | Values                                                 | Description                                                                                                            |
|-------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `FIDESLIB_ARCH`               | **"<all supported>"**, `auto`, string                  | Target GPU architectures. `auto` builds only for the GPUs visible via `nvidia-smi`; an explicit list (e.g. `89-real`) overrides. |
| `CMAKE_BUILD_TYPE`            | **"Release"**, "Debug", "MinSizeRel", "RelWithDebInfo" | Select the compilation build type.                                                                                     |
| `FIDESLIB_INSTALL_PREFIX`     | **"/usr/local"**,string                                | Select prefix path for the installation path of FIDESlib. Relative paths are resolved from the project root directory. |
| `OPENFHE_INSTALL_PREFIX`      | **"/usr/local"**,string                                | Select prefix path for the installation path of OpenFHE. Relative paths are resolved from the project root directory.  |
| `FIDESLIB_INSTALL_OPENFHE`    | ON / **OFF**                                           | Enable the automatic installation of the patched OpenFHE 1.6.1. Needed the first time.                          |
| `FIDESLIB_OPENFHE_SOURCE`     | **"patch"**, "patch-minimal", "port"                  | How the auto-installer obtains OpenFHE (see the table above).                                                          |
| `FIDESLIB_COMPILE_TESTS`      | **ON** / OFF                                           | Build the tests for verifying the functionality of the project.                                                        |
| `FIDESLIB_COMPILE_BENCHMARKS` | **ON** / OFF                                           | Build the benchmarks executable.                                                                                       |
| `FIDESLIB_ENABLE_COVERAGE`    | ON / **OFF**                                           | Build with gcov host-side coverage instrumentation (`scripts/run_coverage.sh`).                                        |
| `BUILD_STATIC`                | **ON** / OFF                                           | Also build static versions of the FIDESlib library.                                                                    |

### FIDESlib Installation

Installing the library is as easy as running the following command:

```bash
cmake --build $PATH_TO_BUILD_DIR --target install -j
```

FIDESlib is currently ready to be consumed as a CMake library. The template project on the examples directory shows how
to build and run a FIDESlib client application and contains examples of usage of most of the functionality provided by
FIDESlib.

> [!NOTE]
> As the default installation prefix for FIDESlib is /usr/local you may need administrator privileges. Change this
> using the previously mentioned configuration options.

## Usage

Check examples for projects that use FIDESlib.

## Coverage

The repository ships a local coverage pipeline (`scripts/run_coverage.sh`) that measures how much of the
library is executed while the test suite runs. Current figures and what they do/do not include are
tracked in [COVERAGE.md](COVERAGE.md).

Requirements:

- A working CUDA environment (the tests need a GPU).
- `gcovr`: `pip3 install gcovr`
- `codecov-cli` (only to upload reports to Codecov): `pip3 install codecov-cli`

Run the pipeline (configures a `build-coverage/` directory, builds the tests, runs them and generates
HTML + Cobertura reports under `coverage/`):

```bash
./scripts/run_coverage.sh
```

The suite is large; a single long process may exceed per-job memory/time limits
(and one crashing test aborts the whole measurement). An alternative sharded
driver, `scripts/run_coverage_sharded.sh`, launches **each test case in its own
process**, accumulates gcov counters across all of them, is crash-isolated and
resumable, and records per-case RSS under `coverage/.stats/`:

```bash
./scripts/run_coverage_sharded.sh
# limit or resume with a subset:
TEST_FILTER='OpenFHEInterfaceTests*:OpenFHEBootstrapTests*' ./scripts/run_coverage_sharded.sh
```

To fail the run when line coverage is below a target, or to upload the report to Codecov:

```bash
COVERAGE_MIN_LINE=60 ./scripts/run_coverage.sh
CODECOV_TOKEN=<upload token> ./scripts/run_coverage.sh
```

Additional environment variables (all optional): `FIDESLIB_ARCH` (pin CUDA architectures, e.g. `89-real`
or `auto`; unset = portable list that runs on any GPU), `OPENFHE_INSTALL_PREFIX` (non-default OpenFHE
install), `BUILD_DIR` (default `build-coverage`), `REPORT_DIR` (default `coverage`), `JOBS`,
`TEST_FILTER` (gtest filter, e.g. `NTTTests/NTTTest.*` or `:AccumulateBroadcast.*`; empty = the whole suite).

Coverage is collected with gcov (`--coverage`, enabled through the `FIDESLIB_ENABLE_COVERAGE` CMake
option) and reported with gcovr. Only `src/` and `api/` are measured; tests, benchmarks and
third-party code are excluded.

> [!WARNING]
> gcov instruments the **host** side of `.cpp`/`.cu` files. CUDA **device** code
> (`__device__`/`__global__` bodies) is not instrumented and will show up as uncovered, so
> kernel-heavy files report low percentages even when they are well tested. Treat the numbers as
> host-code coverage.

## Docker

Check docker directory to obtain instructions on running FIDESlib inside a Docker environment.

## Credits

Thanks to all main contributors:

* Carlos Agulló Domingo.
* Óscar Vera López.
* Seyda Guzelhan.
* Lohit Daksha.
* Aymane El Jerari.

And thanks to our advisor:

* José L. Abellán.

## Grants

This project was possible thanks to the following grants:

* Grant CNS2023-144241 funded by "MICIU/AEI/10.13039/501100011033" and the "European Union NextGenerationEU/PRTR".
* Grants NSF CNS 2312275 and 2312276, and supported in part from the NSF IUCRC Center for Hardware and Embedded Systems
  Security and Trust (CHEST).

## Inquiries and comments

If you have any question, comment, or suggestion, please contact:

* Carlos Agulló Domingo (carlos.a.d@um.es).
* Óscar Vera López (oscar.veral@um.es).

Or feel free to open an issue or a general discussion on this repository.
