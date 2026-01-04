# CMakeLists.txt for C++ Materials Database API
cmake_minimum_required(VERSION 3.15)
project(materials_database_cpp
    VERSION 1.0.0
    DESCRIPTION "C++ API for thermal materials database with depth-varying properties"
    LANGUAGES CXX
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test suite" OFF)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Library target - build as shared library
add_library(materials_database
    src/materials_db.cpp
)

# Add alias for consistency with installed target
add_library(materials::materials_database ALIAS materials_database)

# Set library properties
set_target_properties(materials_database PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "materials_db"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(materials_database
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(materials_database
    PUBLIC
        SQLite::SQLite3
)

# Compiler warnings
if(MSVC)
    target_compile_options(materials_database PRIVATE /W4)
else()
    target_compile_options(materials_database PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(example_usage
        examples/example_usage.cpp
    )

    target_link_libraries(example_usage
        PRIVATE
            materials_database
    )

    add_executable(example_accessors
        examples/example_accessors.cpp
    )

    target_link_libraries(example_accessors
        PRIVATE
            materials_database
    )
endif()

# Installation
include(GNUInstallDirs)

# Install library and create export set
install(TARGETS materials_database
    EXPORT materials_database_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
        NAMELINK_COMPONENT Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
)

# Install export set - allows downstream projects to use find_package()
install(EXPORT materials_database_targets
    FILE materials_database_targets.cmake
    NAMESPACE materials::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/materials_database
    COMPONENT Development
)

# Package configuration
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/materials_database_config_version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/materials_database_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/materials_database_config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/materials_database
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/materials_database_config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/materials_database_config_version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/materials_database
    COMPONENT Development
)

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Materials Database C++ API Configuration")
message(STATUS "========================================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type:      ${BUILD_SHARED_LIBS}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  BUILD_EXAMPLES:    ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_TESTS:       ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  SQLite3:           ${SQLite3_VERSION}")
message(STATUS "")
message(STATUS "Install directories:")
message(STATUS "  Libraries:         ${CMAKE_INSTALL_FULL_LIBDIR}")
message(STATUS "  Headers:           ${CMAKE_INSTALL_FULL_INCLUDEDIR}")
message(STATUS "  CMake config:      ${CMAKE_INSTALL_FULL_LIBDIR}/cmake/materials_database")
message(STATUS "========================================")
message(STATUS "")
