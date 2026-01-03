# C++ Materials Database API

A modern C++17 API for accessing the SQLite materials database with depth-varying thermal properties.

## Features

- **Modern C++17** - Uses `std::optional`, `std::vector`, exceptions
- **Type-safe** - Strong typing for all material properties
- **Header + Implementation** - Clean separation for easy integration
- **Full CRUD operations** - Create, Read, Update, Delete materials
- **Depth interpolation** - Linear interpolation to arbitrary depths
- **SQLite3 backend** - Industry-standard database
- **CMake build system** - Easy integration into existing projects
- **Comprehensive examples** - Working code to get started quickly

## Quick Start

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or later
- SQLite3 library

### Building

```bash
cd cpp_api
mkdir build && cd build
cmake ..
cmake --build .
```

### Running the Example

```bash
# From build directory
./example_usage
```

## Basic Usage

### Opening a Database

```cpp
#include "materials_db.hpp"

using namespace thermal;

// Open existing database
MaterialDatabase db("data/materials/materials.db");

// Or create new database if it doesn't exist
MaterialDatabase db("my_materials.db", true);
```

### Querying Materials

```cpp
// Query by name (case-insensitive)
auto mat = db.get_material_by_name("Desert Sand (Depth-Varying)");

if (mat) {
    std::cout << "Material: " << mat->name << '\n';
    std::cout << "k range: " << mat->k.front() << " - " << mat->k.back() << " W/(m·K)\n";

    // Compute derived properties
    double I = mat->thermal_inertia(0.0);  // Surface thermal inertia
    double alpha = mat->thermal_diffusivity(0.0);  // Surface diffusivity

    std::cout << "Thermal inertia: " << I << " J/(m²·K·s^0.5)\n";
}
```

### Interpolating to Solver Grid

```cpp
// Define solver grid depths
std::vector<double> target_depths = {0.0, 0.05, 0.10, 0.20, 0.50};

// Interpolate properties
std::vector<double> k_interp, rho_interp, cp_interp;
mat->interpolate_to_depths(target_depths, k_interp, rho_interp, cp_interp);

// Use in your thermal solver
for (size_t i = 0; i < target_depths.size(); ++i) {
    double k = k_interp[i];
    double rho = rho_interp[i];
    double cp = cp_interp[i];
    // ... pass to solver
}
```

### Listing All Materials

```cpp
auto materials = db.list_all_materials();

for (const auto& name : materials) {
    std::cout << name << '\n';
}
```

### Adding New Materials

```cpp
MaterialPropertiesDepth new_mat;

// Basic info
new_mat.name = "Custom Material";
new_mat.version = 1;

// Depth-varying thermal properties
new_mat.depths = {0.0, 0.10, 0.25, 0.50};
new_mat.k = {0.5, 0.6, 0.7, 0.8};        // W/(m·K)
new_mat.rho = {1600, 1650, 1700, 1750};   // kg/m³
new_mat.cp = {850, 850, 850, 850};        // J/(kg·K)

// Surface radiative properties
new_mat.alpha = 0.85;      // Solar absorptivity
new_mat.epsilon = 0.90;    // Thermal emissivity
new_mat.roughness = 0.005; // Surface roughness [m]

// Provenance
new_mat.source_database = "My Database";
new_mat.source_citation = "Smith et al. (2026)";
new_mat.notes = "Measured in laboratory";

// Add to database
std::string mat_id = db.add_material(new_mat);
std::cout << "Added material with ID: " << mat_id << '\n';
```

## API Reference

### Classes

#### `MaterialPropertiesDepth`

Represents a material with depth-varying thermal properties.

**Public Members:**
```cpp
std::string material_id;          // UUID
std::string name;                 // Material name
int version;                      // Version number

// Depth-varying thermal properties
std::vector<double> depths;       // Depth values [m]
std::vector<double> k;            // Thermal conductivity [W/(m·K)]
std::vector<double> rho;          // Density [kg/m³]
std::vector<double> cp;           // Specific heat [J/(kg·K)]

// Surface properties
double alpha;                     // Solar absorptivity [0-1]
double epsilon;                   // Thermal emissivity [0-1]
double roughness;                 // Surface roughness [m]

// Provenance
std::string source_database;
std::string source_citation;
std::string notes;
std::string supersedes;           // UUID of superseded material
```

**Methods:**
```cpp
// Compute thermal diffusivity at depth
double thermal_diffusivity(double depth_m = 0.0) const;

// Compute thermal inertia at depth
double thermal_inertia(double depth_m = 0.0) const;

// Interpolate individual properties
double interpolate_k(double depth_m) const;
double interpolate_rho(double depth_m) const;
double interpolate_cp(double depth_m) const;

// Interpolate all properties to multiple depths
void interpolate_to_depths(
    const std::vector<double>& target_depths,
    std::vector<double>& k_out,
    std::vector<double>& rho_out,
    std::vector<double>& cp_out
) const;

// Validate material properties
void validate() const;
```

#### `MaterialDatabase`

SQLite database interface for materials.

**Constructor:**
```cpp
MaterialDatabase(const std::string& db_path, bool create_if_missing = false);
```

**Query Methods:**
```cpp
// Get material by UUID
std::optional<MaterialPropertiesDepth> get_material(const std::string& material_id) const;

// Get material by name (case-insensitive)
std::optional<MaterialPropertiesDepth> get_material_by_name(const std::string& name) const;

// List all material names
std::vector<std::string> list_all_materials() const;

// Get database statistics
void get_statistics(int& total_materials, int& depth_varying_count) const;
```

**Modification Methods:**
```cpp
// Add new material (returns UUID)
std::string add_material(const MaterialPropertiesDepth& material);

// Update existing material
void update_material(const MaterialPropertiesDepth& material);

// Delete material by ID
void delete_material(const std::string& material_id);
```

**Database Management:**
```cpp
// Close database connection
void close();

// Check if database is open
bool is_open() const;

// Create database schema
void create_schema();

// Verify database integrity
bool verify_integrity() const;
```

### Exception Handling

The API uses exceptions for error reporting:

```cpp
try {
    MaterialDatabase db("materials.db");
    auto mat = db.get_material_by_name("Nonexistent Material");
    // ...
} catch (const DatabaseError& e) {
    std::cerr << "Database error: " << e.what() << '\n';
}
```

## Integration with Your Project

### Method 1: CMake Subdirectory

Add to your `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/cpp_api)
target_link_libraries(your_target PRIVATE materials_database)
```

### Method 2: CMake Install

```bash
cd cpp_api/build
cmake --install . --prefix /usr/local
```

Then in your project:

```cmake
find_package(materials_database REQUIRED)
target_link_libraries(your_target PRIVATE materials::materials_database)
```

### Method 3: Direct Source

Simply copy `include/materials_db.hpp` and `src/materials_db.cpp` into your project and compile.

## Database Schema

The SQLite database uses five tables:

**materials**: Metadata, versioning, provenance
```sql
CREATE TABLE materials (
    material_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    version INTEGER NOT NULL DEFAULT 1,
    source_database TEXT,
    source_citation TEXT,
    notes TEXT,
    supersedes TEXT,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**thermal_properties**: Depth-varying k, ρ, cp
```sql
CREATE TABLE thermal_properties (
    material_id TEXT NOT NULL,
    depth_m REAL NOT NULL,
    thermal_conductivity REAL NOT NULL,
    density REAL NOT NULL,
    specific_heat REAL NOT NULL,
    PRIMARY KEY (material_id, depth_m)
);
```

**radiative_properties**: Surface α, ε
```sql
CREATE TABLE radiative_properties (
    material_id TEXT PRIMARY KEY,
    solar_absorptivity REAL NOT NULL,
    thermal_emissivity REAL NOT NULL
);
```

**surface_properties**: Roughness
```sql
CREATE TABLE surface_properties (
    material_id TEXT PRIMARY KEY,
    roughness REAL NOT NULL
);
```

**spectral_emissivity**: Wavelength-dependent ε (future use)
```sql
CREATE TABLE spectral_emissivity (
    material_id TEXT NOT NULL,
    wavelength_um REAL NOT NULL,
    emissivity REAL NOT NULL,
    PRIMARY KEY (material_id, wavelength_um)
);
```

## Performance Considerations

- **Database connections**: Reuse `MaterialDatabase` objects instead of creating new ones
- **Interpolation**: Pre-compute interpolated values rather than calling `interpolate_*()` in tight loops
- **Transactions**: For bulk inserts, wrap in transactions (call `execute("BEGIN")` ... `execute("COMMIT")`)
- **Memory**: `MaterialPropertiesDepth` uses `std::vector` which allocates on heap - consider object pooling for high-frequency allocations

## Examples

See `examples/example_usage.cpp` for comprehensive working examples including:

1. Querying materials by name
2. Interpolating properties to solver grids
3. Listing all materials
4. Getting database statistics
5. Adding new materials

## Comparison with Python API

The C++ API mirrors the Python API functionality:

| Python | C++ |
|--------|-----|
| `MaterialDatabaseSQLite` | `MaterialDatabase` |
| `MaterialPropertiesDepth` | `MaterialPropertiesDepth` |
| `get_material_by_name()` | `get_material_by_name()` returns `std::optional` |
| `interpolate_to_depths()` | `interpolate_to_depths()` with output parameters |
| Exception-based errors | Exception-based errors (`DatabaseError`) |

## License

Same as main thermal terrain simulator project.

## Contributing

Bug reports and pull requests welcome. Please maintain:
- C++17 compatibility
- Consistent coding style
- Comprehensive error checking
- Doxygen-compatible comments

## Contact

See main project README for contact information.
