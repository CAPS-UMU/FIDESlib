# Pin the clang-format version to minimse cross-version churn (see .clang-format).
# Prefer the versions 17/18+ that are known to match the repo config; fall back to any
# `clang-format` so the targets stay usable on machines with only the distro package.
find_program(CLANG_FORMAT_EXE NAMES clang-format-18 clang-format-17 clang-format-16 clang-format)

if(CLANG_FORMAT_EXE)
    message(STATUS "Found clang-format: ${CLANG_FORMAT_EXE}")

    # Find all source files for format
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.hpp"
    )

    if(ALL_CXX_SOURCE_FILES)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_CXX_SOURCE_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-format on all source files"
            VERBATIM
        )

        # Read-only check: fails the build/CI on any file that is not formatted.
        # NOTE: an unparseable "style=file" config makes clang-format silently fall back to
        # default LLVM style, so validate the config first (see .clang-format instructions).
        add_custom_target(format-check
            COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror -style=file ${ALL_CXX_SOURCE_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Checking clang-format compliance"
            VERBATIM
        )
    else()
        message(STATUS "No source files found to format.")
    endif()
else()
    message(WARNING "clang-format not found! The 'format' and 'format-check' targets will not be available.")
endif()
