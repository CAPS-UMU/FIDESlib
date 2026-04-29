cmake_minimum_required(VERSION 3.25)

############### 1. Setup Project & CUDA ###############
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Setup CUDA Path
if (NOT DEFINED CUDA_PATH)
    set(CUDA_PATH "/usr/local/cuda")
endif ()
set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")

if (NOT EXISTS "${CUDA_PATH}")
    message(FATAL_ERROR "CUDA not found at ${CUDA_PATH}. Please install CUDA.")
endif ()

set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90-real;90-virtual;100-real;120-real")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

project(llm)

############### 2. PATH DEFINITIONS ###############

set(MY_OPENFHE_ROOT "../../deps/openfhe-install")
set(MY_OPENFHE_INC "${MY_OPENFHE_ROOT}/include/openfhe")
set(MY_OPENFHE_LIB "${MY_OPENFHE_ROOT}/lib")

link_directories(${MY_OPENFHE_LIB})

############### 3. Packages & Includes ###############


find_package(fideslib REQUIRED CONFIG)

# FIDESlib configuration.

if (fideslib_FOUND)
    message(STATUS "FOUND PACKAGE FIDESlib")
    message(STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message(STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message(STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
    message(STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
else ()
    message(FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

if (NOT DEFINED CMAKE_CXX_COMPILER OR NOT CMAKE_CXX_COMPILER STREQUAL FIDESLIB_CXX_COMPILER)
    # Align host compiler with fideslib toolchain to avoid ABI mismatches.
    set(CMAKE_CXX_COMPILER "${FIDESLIB_CXX_COMPILER}" CACHE STRING "C++ compiler" FORCE)
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER OR NOT CMAKE_CUDA_HOST_COMPILER STREQUAL FIDESLIB_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${FIDESLIB_CXX_COMPILER}" CACHE STRING "CUDA host compiler" FORCE)
endif ()

enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(${OPENMP_INCLUDES})
include_directories(${MY_OPENFHE_INC})
include_directories(${MY_OPENFHE_INC}/pke)
include_directories(${MY_OPENFHE_INC}/core)
include_directories(${MY_OPENFHE_INC}/binfhe)
include_directories(${MY_OPENFHE_INC}/..)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use Clang's OpenMP implementation.
    set(FIX_OMP_FLAGS "-fopenmp=libomp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(FIX_OMP_FLAGS "-fopenmp")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIX_OMP_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${FIX_OMP_FLAGS}")

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT supported OUTPUT error)

    if (supported)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(STATUS "IPO / LTO not supported: <${error}>")
    endif ()
endif ()

############### 4. Sources & Helper Lib ###############

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
message(STATUS "Source dir:  ${SOURCE_DIR}")

file(GLOB_RECURSE CXX_SOURCE_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES ${SOURCE_DIR}/*.cu ${SOURCE_DIR}/*.cuh)

set_source_files_properties(${CXX_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

# --- FIX 2: Define Absolute Path Variables for Direct Linking ---
set(OPENFHE_STATIC_LIBS
        #        "${MY_OPENFHE_LIB}/libOPENFHEpke_static.a"
        #        "${MY_OPENFHE_LIB}/libOPENFHEbinfhe_static.a"
        #        "${MY_OPENFHE_LIB}/libOPENFHEcore_static.a"
)

# Create Static Helper Library 'llm'
add_library(llm STATIC ${CXX_SOURCE_FILES} ${CUDA_SOURCE_FILES})

# Ensure gtest headers are visible to llm sources and consumers
#target_include_directories(llm
#PUBLIC
#        ${gtest_SOURCE_DIR}/googletest/include
#        ${gtest_BINARY_DIR}/googletest/include
#)


target_link_libraries(llm PRIVATE fideslib::fideslib)
#target_link_libraries(llm PUBLIC ${OPENFHE_STATIC_LIBS})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
#  --generate-code=arch=compute_80,code=[compute_80,lto_80,sm_80]")

target_compile_options(llm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall -rdc=true -diag-suppress 611 >
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -g>
)
target_link_options(llm PUBLIC
        $<DEVICE_LINK:SHELL:-dlto>
)
set_target_properties(llm PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
)


############### 5. Executables ###############

# berttiny-all
add_executable(berttiny-all main/berttiny-all.cu)
target_include_directories(berttiny-all PRIVATE ${SOURCE_DIR})
target_link_libraries(berttiny-all PRIVATE fideslib::fideslib llm)
target_link_libraries(berttiny-all PRIVATE ${OPENFHE_STATIC_LIBS})

set_target_properties(berttiny-all PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# berttiny-simple
#add_executable(berttiny-simple main/berttiny-simple.cu)
#target_include_directories(berttiny-simple PRIVATE ${SOURCE_DIR})
#target_link_libraries(berttiny-simple PRIVATE fideslib::fideslib llm)
#target_link_libraries(berttiny-simple PRIVATE ${OPENFHE_STATIC_LIBS})

#set_target_properties(berttiny-simple PROPERTIES
#        CUDA_RESOLVE_DEVICE_SYMBOLS ON
#        CUDA_SEPARABLE_COMPILATION ON
#        POSITION_INDEPENDENT_CODE ON
#)


# GoogleTest for example-side unit tests
set(INSTALL_GTEST OFF)
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# berttiny-test
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)
file(GLOB_RECURSE TEST_SOURCE_FILES ${TEST_DIR}/*.cu ${TEST_DIR}/*.cuh)

add_executable(berttiny-test ${TEST_SOURCE_FILES})
target_include_directories(berttiny-test PRIVATE ${SOURCE_DIR} ${TEST_DIR})
target_link_libraries(berttiny-test PRIVATE fideslib::fideslib llm)
target_link_libraries(berttiny-test PRIVATE ${OPENFHE_STATIC_LIBS} GTest::gtest GTest::gtest_main)

set_target_properties(berttiny-test PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

add_test(NAME berttiny-test COMMAND berttiny-test)