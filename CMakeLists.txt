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

if(NOT CMAKE_GENERATOR MATCHES "Ninja|Visual Studio 17 2022")
    message(FATAL_ERROR
        "Spar's C++ module build requires Ninja, Ninja Multi-Config, or Visual Studio 17 2022. "
        "Detected generator: ${CMAKE_GENERATOR}.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
        message(FATAL_ERROR
            "Spar requires LLVM Clang 18 or newer for reliable C++ module support. "
            "Detected ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
        message(FATAL_ERROR "Spar requires GCC 14 or newer for C++ module support.")
    endif()
elseif(NOT MSVC)
    message(WARNING
        "This compiler has not yet been validated for Spar's C++ module build. "
        "LLVM Clang 18+ with Ninja is the supported path.")
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
        src/core/shape.cpp
        src/core/tensor.cpp
        src/core/random.cpp
    PUBLIC
        FILE_SET spar_modules TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
        FILES
            src/spar.cppm
            src/core/dtype.cppm
            src/core/shape.cppm
            src/core/tensor.cppm
            src/core/random.cppm
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
