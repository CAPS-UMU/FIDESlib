<p align="center">
  <img src="https://github.com/CAPS-UMU/FIDESlib/blob/main/doxygen/FidesLogo.drawio.svg?raw=true" width="200">
</p>

# FIDESlib 2.1.2

A server-side CKKS GPU library fully interoperable with OpenFHE.

## Improvements in version 2.1.2
  - OpenFHE version 1.5.1 compatibility.
  - Multi-GPU support with NCCL.
  - Sparse Secret Encapsulation support.
  - Many performance optimizations.
  - Bug fixes.

## Features
  -  Full CKKS implementation: Add, AddPt, AddScalar, Mult, MultPt, MultScalar, Square, Rotate, RotateHoisted, Bootstrap.
  -  OpenFHE interoperability for FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO and FLEXIBLEAUTOEXT.
  -  Hardware acceleration with NVIDIA CUDA.
  -  High-performance NTT/INTT implementation.
  -  Hybrid Key-Switching.

## Citation

If you use FIDESlib on your research, please cite our ISPASS paper.

```bibtex
@inproceedings{FIDESlib,
  author    = {Carlos Agulló-Domingo and Óscar Vera-López and Seyda Guzelhan and Lohit Daksha and Aymane El Jerari and Kaustubh Shivdikar and Rashmi Agrawal and David Kaeli and Ajay Joshi and José L. Abellán},
  title     = {{FIDESlib: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs}},
  booktitle = {2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year      = {2025},
  note      = {Poster paper},
  url       = {https://github.com/CAPS-UMU/FIDESlib},
  publisher = {IEEE},
  address = {Ghent, Belgium},
}
```

## Compilation

> [!IMPORTANT]
> Requirements:
>  -  NVIDIA CUDA version 12 or 13.
>  -  GCC version >=11
>  -  OpenMP development library.
>  -  CMake version 3.25.2 or greater.
>  -  (Optional) NVIDIA Collective Communications Library to enable Multi-GPU support.

### Requirements installation

For CUDA software stack, follow the official installation guides. For the remaining
dependencies, install them using the package manager or mehtod of your choice. 

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

FIDESlib needs a patched version of OpenFHE in order to be able to access some internals needed for interoperability. This patched version can be automatically installed by defining FIDESLIB_INSTALL_OPENFHE=ON CMake variable. By default this variable is set OFF. Once the patched version is installed, one can disable this flag when reinstalling FIDESlib.

The build process produces the following artifacts: 
- fideslib.so: The FIDESlib library to be dynamically linked to any client application.
- fideslib-test: The test suite executable if selected.
- fideslib-bench: The benchmark suite executable if selected.
- gpu-test: A dummy executable to search for the CUDA capable devices on the machine.
- dummy: Another dummy executable.

The following options can be used with CMake to configure the build. The default value for each option is denoted in **boldface** under the **Values** column:

| CMake Option                  | Values              | Description |
|-------------------------------|---------------------|------------------------------------------------------
| `FIDESLIB_ARCH`               | **"all-major"**,string | Select the GPU architectures of the selected backend. |
| `CMAKE_BUILD_TYPE`            | **"Release"**, "Debug", "MinSizeRel", "RelWithDebInfo" | Select the compilation build type. |
| `FIDESLIB_INSTALL_PREFIX`     | **"/usr/local"**,string | Select prefix path for the installation path of FIDESlib. Relative paths are resolved from the project root directory. |
| `OPENFHE_INSTALL_PREFIX`      | **"/usr/local"**,string | Select prefix path for the installation path of OpenFHE. Relative paths are resolved from the project root directory. |
| `FIDESLIB_INSTALL_OPENFHE`    | ON / **OFF**        | Enable the installation of the patched version of OpenFHE. Needed the first time. |
| `FIDESLIB_COMPILE_TESTS`      | **ON** / OFF        | Build the tests for verifying the functionality of the project. |
| `FIDESLIB_COMPILE_BENCHMARKS` | **ON** / OFF        | Build the benchmarks executable. |

### FIDESlib Installation

Installing the library is as easy as running the following command:

```bash
cmake --build $PATH_TO_BUILD_DIR --target install -j
```

FIDESlib is currently ready to be consumed as a CMake library. The template project on the examples directory shows how to build and run a FIDESlib client application and contains examples of usage of most of the functionality provided by FIDESlib.

> [!NOTE]
> As the default installation prefix for FIDESlib is /usr/local you may need administrator priviledges. Change this using the previously mentioned configuration options.

## Usage

Check examples for projects that use FIDESlib.

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
* Grants NSF CNS 2312275 and 2312276, and supported in part from the NSF IUCRC Center for Hardware and Embedded Systems Security and Trust (CHEST).

## Inquiries and comments

If you have any question, comment, or suggestion, please contact:
* Carlos Agulló Domingo (carlos.a.d@um.es).
* Óscar Vera López (oscar.veral@um.es).

Or feel free to open an issue or a general discussion on this repository.
