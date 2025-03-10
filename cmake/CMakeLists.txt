cmake_minimum_required (VERSION 3.28 FATAL_ERROR)
cmake_policy (VERSION 3.21.3...3.28)

# Project to build and install OpenFHE at configure time.
project(OpenFHE)

include(ExternalProject)

set(PATCH_HOOK "${CMAKE_CURRENT_SOURCE_DIR}/openfhe-hook.patch")
set(PATCH_BASE "${CMAKE_CURRENT_SOURCE_DIR}/openfhe-base.patch")

# Export compile commands for debugging.
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

ExternalProject_Add(OpenFHE
	GIT_REPOSITORY https://github.com/openfheorg/openfhe-development.git
	GIT_TAG main
	PATCH_COMMAND git reset --hard v1.2.3 && git config user.email "FIDESlib" && git config user.name "FIDESlib" && git am ${PATCH_HOOK} && git am ${PATCH_BASE}
)