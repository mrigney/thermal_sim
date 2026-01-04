# CMake Package Configuration

This directory contains CMake configuration files that enable the Materials Database library to be easily consumed by downstream projects using `find_package()`.

## For Library Users

After installing the library (see main README.md), you can use it in your CMake project:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_project)

# Find the installed library
find_package(materials_database 1.0 REQUIRED)

# Link against it
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE materials::materials_database)
```

The `find_package()` call will:
1. Locate the installed library
2. Find SQLite3 dependency automatically
3. Set up all necessary include paths and link libraries
4. Provide the `materials::materials_database` target

## Configuration Files

- **materials_database_config.cmake.in**: Template for the package configuration file
  - Finds required dependencies (SQLite3)
  - Includes the exported targets file
  - Validates that all components are present

- **materials_database_targets.cmake**: Auto-generated during installation
  - Contains the actual imported target definitions
  - Created by CMake's export mechanism

- **materials_database_config_version.cmake**: Auto-generated version file
  - Enables version checking in `find_package()`
  - Uses `SameMajorVersion` compatibility

## Example: Finding with Specific Version

```cmake
# Require exact version
find_package(materials_database 1.0.0 EXACT REQUIRED)

# Require minimum version
find_package(materials_database 1.0 REQUIRED)

# Optional dependency
find_package(materials_database 1.0 QUIET)
if(materials_database_FOUND)
    # Use the library
endif()
```

## Install Location

After installation, the config files are located at:
```
${CMAKE_INSTALL_PREFIX}/lib/cmake/materials_database/
  ├── materials_database_config.cmake
  ├── materials_database_config_version.cmake
  └── materials_database_targets.cmake
```

On Linux, this is typically `/usr/local/lib/cmake/materials_database/`.
On Windows, this depends on your CMAKE_INSTALL_PREFIX.
