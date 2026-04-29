find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format-18 clang-format-17 clang-format-16 clang-format-15)

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
    else()
        message(STATUS "No source files found to format.")
    endif()
else()
    message(WARNING "clang-format not found! The 'format' target will not be available.")
endif()
