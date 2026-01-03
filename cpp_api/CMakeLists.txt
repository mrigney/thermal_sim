# CMakeLists.txt for C++ Materials Database API
cmake_minimum_required(VERSION 3.15)
project(materials_database_cpp VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test suite" OFF)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Library target
add_library(materials_database
    src/materials_db.cpp
)

target_include_directories(materials_database
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${SQLITE3_INCLUDE_DIRS}
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
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS materials_database
    EXPORT materials_database_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT materials_database_targets
    FILE materials_database_targets.cmake
    NAMESPACE materials::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/materials_database
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
)
