# Using the Installed Library in Your Project

This guide shows how to use the Materials Database library in your own CMake project after it has been installed.

## Example Project Structure

```
my_thermal_app/
├── CMakeLists.txt
└── src/
    └── main.cpp
```

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_thermal_app VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed materials database library
# This automatically finds SQLite3 as well
find_package(materials_database 1.0 REQUIRED)

# Create your application
add_executable(my_thermal_app src/main.cpp)

# Link against the materials database
# This also links SQLite3 and sets up include paths
target_link_libraries(my_thermal_app PRIVATE materials::materials_database)
```

## src/main.cpp

```cpp
#include <iostream>
#include "materials_db.hpp"

int main() {
    try {
        // Open the materials database
        thermal::MaterialDatabase db("materials.db");

        // Query a material
        auto mat = db.get_material_by_name("Lunar Regolith");

        if (mat) {
            std::cout << "Loaded material: " << mat->name << "\n";
            std::cout << "Thermal conductivity at surface: "
                      << mat->k.front() << " W/(m·K)\n";

            // Compute thermal inertia
            double inertia = mat->thermal_inertia(0.0);
            std::cout << "Thermal inertia: " << inertia << " J/(m²·K·s^0.5)\n";
        } else {
            std::cout << "Material not found\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

## Building Your Project

```bash
# Configure and build
mkdir build && cd build
cmake ..
cmake --build .

# Run
./my_thermal_app
```

## If Library is Installed in Non-Standard Location

If you installed the library to a custom prefix (e.g., `/opt/materials_db`):

```bash
# Option 1: Set CMAKE_PREFIX_PATH
cmake -DCMAKE_PREFIX_PATH=/opt/materials_db ..

# Option 2: Set materials_database_DIR directly
cmake -Dmaterials_database_DIR=/opt/materials_db/lib/cmake/materials_database ..
```

## Advanced Usage

### Using Specific Version

```cmake
# Require exact version
find_package(materials_database 1.0.0 EXACT REQUIRED)

# Require minimum version
find_package(materials_database 1.0 REQUIRED)
```

### Optional Dependency

```cmake
find_package(materials_database 1.0 QUIET)

if(materials_database_FOUND)
    target_link_libraries(my_app PRIVATE materials::materials_database)
    target_compile_definitions(my_app PRIVATE HAVE_MATERIALS_DB)
else()
    message(STATUS "Materials database not found - using fallback")
endif()
```

### Linking Multiple Targets

```cmake
add_executable(app1 src/app1.cpp)
add_executable(app2 src/app2.cpp)
add_library(mylib src/mylib.cpp)

# All get the materials database
target_link_libraries(app1 PRIVATE materials::materials_database)
target_link_libraries(app2 PRIVATE materials::materials_database)
target_link_libraries(mylib PUBLIC materials::materials_database)
```

## What Gets Linked Automatically

When you link against `materials::materials_database`, CMake automatically:

1. **Links the shared library**: `libmaterials_db.so` (or platform equivalent)
2. **Links SQLite3**: Transitive dependency from the library
3. **Adds include directories**: Headers from `include/` directory
4. **Sets proper RPATH**: On Unix systems, for finding the shared library at runtime

No manual include paths or library paths needed!

## Troubleshooting

### "Could not find a package configuration file"

The library isn't installed or CMake can't find it. Either:
- Install the library first
- Set `CMAKE_PREFIX_PATH` to the install location
- Set `materials_database_DIR` to `<install>/lib/cmake/materials_database`

### "undefined reference to" linker errors

Make sure you're linking against `materials::materials_database` (with the namespace), not just `materials_database`.

### SQLite3 not found

Install SQLite3 development files:
- Ubuntu/Debian: `sudo apt-get install libsqlite3-dev`
- Fedora/RHEL: `sudo dnf install sqlite-devel`
- macOS: `brew install sqlite3`
- Windows: Download from https://www.sqlite.org/download.html
