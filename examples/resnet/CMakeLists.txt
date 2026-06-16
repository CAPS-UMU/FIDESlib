cmake_minimum_required (VERSION 3.25.2)

# C++ Config

set (CMAKE_CXX_STANDARD 20)

# Project

project (resnet LANGUAGES CXX)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages

find_package (fideslib REQUIRED CONFIG)

# FIDESlib configuration.

if (fideslib_FOUND)
    message (STATUS "FOUND PACKAGE FIDESlib")
    message (STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message (STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message (STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message (STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
    message (STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
else ()
    message (FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

set (CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

# Executable definition.

set (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file (GLOB_RECURSE CXX_SOURCE_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.hpp)
set_source_files_properties(${CXX_SOURCE_FILES} PROPERTIES LANGUAGE CXX)

add_executable (resnet)

target_sources(resnet PRIVATE ${CXX_SOURCE_FILES})

# resnet uses #include <fideslib/...> style includes.  Derive the include root from the
# fideslib cmake dir (e.g. <prefix>/share/fideslib/cmake -> <prefix>/include).
get_filename_component(_fideslib_prefix "${fideslib_DIR}/../../.." ABSOLUTE)
target_include_directories(resnet PRIVATE "${_fideslib_prefix}/include")

target_link_libraries (resnet PRIVATE fideslib::fideslib)