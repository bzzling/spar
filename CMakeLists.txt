cmake_minimum_required(VERSION 4.4)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")

if(APPLE)
    if(NOT CMAKE_OSX_SYSROOT)
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            RESULT_VARIABLE SPAR_XCRUN_RESULT
            OUTPUT_VARIABLE SPAR_MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(SPAR_XCRUN_RESULT EQUAL 0)
            set(CMAKE_OSX_SYSROOT "${SPAR_MACOS_SDK_PATH}" CACHE PATH "macOS SDK path")
        endif()
    endif()

    if(NOT DEFINED CMAKE_CXX_STDLIB_MODULES_JSON)
        foreach(SPAR_LLVM_PREFIX IN ITEMS /opt/homebrew/opt/llvm /usr/local/opt/llvm)
            if(EXISTS "${SPAR_LLVM_PREFIX}/lib/c++/libc++.modules.json")
                set(
                    CMAKE_CXX_STDLIB_MODULES_JSON
                    "${SPAR_LLVM_PREFIX}/lib/c++/libc++.modules.json"
                    CACHE FILEPATH "C++ standard-library module manifest"
                )
                break()
            endif()
        endforeach()
    endif()
endif()

project(
    Spar
    VERSION 0.1.0
    DESCRIPTION "A dependency-free C++23 deep-learning framework"
    LANGUAGES CXX
)

set(CMAKE_CXX_MODULE_STD ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(SPAR_BUILD_TESTS "Build Spar tests" OFF)
option(SPAR_BUILD_EXAMPLES "Build Spar examples" OFF)

if(NOT CMAKE_GENERATOR MATCHES "^Ninja")
    message(FATAL_ERROR
        "Spar's C++23 module build currently requires Ninja or Ninja Multi-Config. "
        "Detected generator: ${CMAKE_GENERATOR}.")
endif()

if(NOT 23 IN_LIST CMAKE_CXX_COMPILER_IMPORT_STD
   OR CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE)
    set(SPAR_IMPORT_STD_DETAIL "")
    if(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE)
        string(APPEND SPAR_IMPORT_STD_DETAIL
            " CMake reported: ${CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE}")
    endif()
    message(FATAL_ERROR
        "Spar requires C++23 'import std' support, but CMake did not detect it for "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} with ${CMAKE_GENERATOR}. "
        "Use a recent LLVM Clang with libc++ standard-library modules and Ninja. "
        "On macOS, set CMAKE_CXX_STDLIB_MODULES_JSON to LLVM libc++'s "
        "libc++.modules.json if it is not found automatically.${SPAR_IMPORT_STD_DETAIL}")
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
   AND NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    message(FATAL_ERROR
        "Spar could not locate libc++.modules.json. Set "
        "CMAKE_CXX_STDLIB_MODULES_JSON to the manifest supplied by LLVM libc++.")
endif()

add_library(spar)
add_library(Spar::Spar ALIAS spar)

target_compile_features(spar PUBLIC cxx_std_23)
set_target_properties(spar PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME Spar
)

target_sources(spar
    PRIVATE
        src/autograd/autograd.cpp
        src/core/storage.cpp
        src/core/shape.cpp
        src/core/tensor.cpp
        src/core/random.cpp
        src/ops/elementwise.cpp
        src/ops/reduction.cpp
        src/ops/matmul.cpp
        src/ops/scalar.cpp
        src/ops/unary.cpp
        src/ops/softmax.cpp
    PUBLIC
        FILE_SET spar_modules TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
        FILES
            src/spar.cppm
            src/core/storage.cppm
            src/core/dtype.cppm
            src/core/shape.cppm
            src/core/tensor.cppm
            src/core/random.cppm
            src/ops/ops.cppm
            src/ops/elementwise.cppm
            src/ops/reduction.cppm
            src/ops/matmul.cppm
            src/ops/scalar.cppm
            src/ops/unary.cppm
            src/ops/softmax.cppm
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(spar PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
elseif(MSVC)
    target_compile_options(spar PRIVATE /W4 /permissive-)
endif()

if(SPAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPAR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

install(TARGETS spar
    EXPORT SparTargets
    FILE_SET spar_modules DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/spar/modules"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SparConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SparConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SparConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Spar"
)

install(EXPORT SparTargets
    FILE SparTargets.cmake
    NAMESPACE Spar::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Spar"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SparConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SparConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Spar"
)
