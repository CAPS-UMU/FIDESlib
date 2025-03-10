cmake_minimum_required (VERSION 3.25.2)

# C++ Config

set (CMAKE_CXX_STANDARD 20)

# Cuda config

set (CMAKE_CUDA_ARCHITECTURES "86-real;89-real")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Project

project (app LANGUAGES CUDA CXX)

# Packages

find_package (FIDESlib REQUIRED CONFIG)
find_package (OpenFHE CONFIG REQUIRED)
find_package (OpenMP REQUIRED)

# OpenFHE configuration.

option (BUILD_STATIC "Set to ON to include static versions of the library" OFF)
if (OpenFHE_FOUND)
    message (STATUS "FOUND PACKAGE OpenFHE")
    message (STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message (STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message (STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message (STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message (STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message (FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set (CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories (${OPENMP_INCLUDES})
include_directories (${OpenFHE_INCLUDE})
include_directories (${OpenFHE_INCLUDE}/third-party/include)
include_directories (${OpenFHE_INCLUDE}/core)
include_directories (${OpenFHE_INCLUDE}/pke)
include_directories (${OpenFHE_INCLUDE}/binfhe)
include_directories (${OpenFHE_INCLUDE}/hook)

link_directories (${OpenFHE_LIBDIR})
link_directories (${OPENMP_LIBRARIES})

if (BUILD_STATIC)
    set (CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries (${OpenFHE_STATIC_LIBRARIES})
else ()
    set (CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries (${OpenFHE_SHARED_LIBRARIES})
endif ()

# FIDESlib configuration.

if (FIDESlib_FOUND)
    message (STATUS "FOUND PACKAGE FIDESlib")
    message (STATUS "FIDESlib Version: ${FIDESlib_VERSION}")
    message (STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message (STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message (STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
else ()
    message (FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

# Executable definition.

set (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file (GLOB_RECURSE CXX_SOURCE_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.hpp)
file (GLOB_RECURSE CUDA_SOURCE_FILES ${SOURCE_DIR}/*.cu ${SOURCE_DIR}/*.cuh)

set_source_files_properties(${CXX_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

add_executable (app)

target_sources(app PRIVATE ${CXX_SOURCE_FILES} ${CUDA_SOURCE_FILES})

target_link_libraries (app PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries (app PRIVATE FIDESlib::fideslib)
