/**
 * @file materials_db.hpp
 * @brief C++ API for SQLite Materials Database
 *
 * Provides C++ interface to the thermal materials database with depth-varying
 * thermal properties. Mirrors the Python API in src/materials_db.py.
 *
 * @author Thermal Terrain Simulator Project
 * @date January 2026
 * @version 1.0
 */

#ifndef MATERIALS_DB_HPP
#define MATERIALS_DB_HPP

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <stdexcept>
#include <cmath>
#include <sqlite3.h>

namespace thermal {

/**
 * @brief Exception class for database errors
 */
class DatabaseError : public std::runtime_error {
public:
    explicit DatabaseError(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Material properties with depth-varying thermal characteristics
 *
 * This class represents a material with thermal properties (k, ρ, cp) that
 * can vary with depth, along with surface radiative and roughness properties.
 *
 * Example:
 * @code
 * MaterialPropertiesDepth mat;
 * mat.name = "Desert Sand";
 * mat.depths = {0.0, 0.05, 0.10, 0.20, 0.50};
 * mat.k = {0.30, 0.35, 0.40, 0.45, 0.50};
 * mat.rho = {1500, 1550, 1600, 1650, 1700};
 * mat.cp = {800, 800, 800, 800, 800};
 * mat.alpha = 0.85;
 * mat.epsilon = 0.90;
 * @endcode
 */
class MaterialPropertiesDepth {
public:
    std::string material_id;     ///< Unique UUID identifier
    std::string name;            ///< Material name
    int version;                 ///< Version number (for updates)

    // Depth-varying thermal properties
    std::vector<double> depths;  ///< Depth values [m] (increasing, starting at 0)
    std::vector<double> k;       ///< Thermal conductivity [W/(m·K)]
    std::vector<double> rho;     ///< Density [kg/m³]
    std::vector<double> cp;      ///< Specific heat capacity [J/(kg·K)]

    // Surface radiative properties
    double alpha;                ///< Solar absorptivity [0-1]
    double epsilon;              ///< Thermal emissivity [0-1]

    // Surface characteristics
    double roughness;            ///< Surface roughness [m]

    // Provenance metadata
    std::string source_database;   ///< Source database name
    std::string source_citation;   ///< Citation for property values
    std::string notes;             ///< Additional notes
    std::string supersedes;        ///< UUID of material this supersedes

    /**
     * @brief Default constructor
     */
    MaterialPropertiesDepth()
        : version(1), alpha(0.0), epsilon(0.0), roughness(0.0) {}

    /**
     * @brief Compute thermal diffusivity at given depth
     * @param depth_m Depth in meters
     * @return Thermal diffusivity α = k/(ρ·cp) [m²/s]
     */
    double thermal_diffusivity(double depth_m = 0.0) const;

    /**
     * @brief Compute thermal inertia at given depth
     * @param depth_m Depth in meters
     * @return Thermal inertia I = √(k·ρ·cp) [J/(m²·K·s^0.5)]
     */
    double thermal_inertia(double depth_m = 0.0) const;

    /**
     * @brief Interpolate thermal conductivity to target depth
     * @param depth_m Target depth [m]
     * @return Interpolated k value [W/(m·K)]
     */
    double interpolate_k(double depth_m) const;

    /**
     * @brief Interpolate density to target depth
     * @param depth_m Target depth [m]
     * @return Interpolated ρ value [kg/m³]
     */
    double interpolate_rho(double depth_m) const;

    /**
     * @brief Interpolate specific heat to target depth
     * @param depth_m Target depth [m]
     * @return Interpolated cp value [J/(kg·K)]
     */
    double interpolate_cp(double depth_m) const;

    /**
     * @brief Interpolate all properties to target depths
     * @param target_depths Vector of target depths [m]
     * @param k_out Output thermal conductivity values
     * @param rho_out Output density values
     * @param cp_out Output specific heat values
     */
    void interpolate_to_depths(
        const std::vector<double>& target_depths,
        std::vector<double>& k_out,
        std::vector<double>& rho_out,
        std::vector<double>& cp_out
    ) const;

    /**
     * @brief Validate material properties
     * @throws DatabaseError if properties are invalid
     */
    void validate() const;

private:
    /**
     * @brief Linear interpolation helper
     * @param x Target x value
     * @param xp X data points (must be sorted ascending)
     * @param fp Y data points
     * @return Interpolated y value
     */
    static double interp1d(double x,
                          const std::vector<double>& xp,
                          const std::vector<double>& fp);
};


/**
 * @brief SQLite database interface for materials with depth-varying properties
 *
 * This class provides CRUD operations for the materials database, mirroring
 * the Python MaterialDatabaseSQLite class.
 *
 * Database schema:
 * - materials: Metadata (name, version, provenance)
 * - thermal_properties: Depth-varying k, ρ, cp
 * - radiative_properties: Surface α, ε
 * - surface_properties: Roughness
 * - spectral_emissivity: Wavelength-dependent ε (future use)
 *
 * Example usage:
 * @code
 * MaterialDatabase db("data/materials/materials.db");
 *
 * // Query by name
 * auto mat = db.get_material_by_name("Desert Sand (Depth-Varying)");
 * if (mat) {
 *     std::cout << "k range: " << mat->k.front() << " - " << mat->k.back() << "\n";
 *     double I = mat->thermal_inertia(0.0);
 *     std::cout << "Thermal inertia (surface): " << I << " J/(m²·K·s^0.5)\n";
 * }
 *
 * // List all materials
 * auto all = db.list_all_materials();
 * for (const auto& name : all) {
 *     std::cout << name << "\n";
 * }
 *
 * db.close();
 * @endcode
 */
class MaterialDatabase {
public:
    /**
     * @brief Open or create materials database
     * @param db_path Path to SQLite database file
     * @param create_if_missing Create database if it doesn't exist (default: false)
     * @throws DatabaseError if database cannot be opened
     */
    explicit MaterialDatabase(const std::string& db_path,
                             bool create_if_missing = false);

    /**
     * @brief Destructor - automatically closes database
     */
    ~MaterialDatabase();

    // Disable copy (sqlite3* should not be copied)
    MaterialDatabase(const MaterialDatabase&) = delete;
    MaterialDatabase& operator=(const MaterialDatabase&) = delete;

    // Enable move
    MaterialDatabase(MaterialDatabase&& other) noexcept;
    MaterialDatabase& operator=(MaterialDatabase&& other) noexcept;

    /**
     * @brief Close database connection
     */
    void close();

    /**
     * @brief Check if database is open
     * @return true if database connection is open
     */
    bool is_open() const { return db_ != nullptr; }

    /**
     * @brief Get material by UUID
     * @param material_id UUID string
     * @return Material properties, or std::nullopt if not found
     */
    std::optional<MaterialPropertiesDepth> get_material(const std::string& material_id) const;

    /**
     * @brief Get material by name (case-insensitive)
     * @param name Material name
     * @return Material properties, or std::nullopt if not found
     */
    std::optional<MaterialPropertiesDepth> get_material_by_name(const std::string& name) const;

    /**
     * @brief Add new material to database
     * @param material Material properties to add
     * @return Material ID (UUID)
     * @throws DatabaseError on database errors
     */
    std::string add_material(const MaterialPropertiesDepth& material);

    /**
     * @brief Update existing material
     * @param material Material properties (must have valid material_id)
     * @throws DatabaseError if material not found or database error
     */
    void update_material(const MaterialPropertiesDepth& material);

    /**
     * @brief Delete material by ID
     * @param material_id UUID string
     * @throws DatabaseError on database errors
     */
    void delete_material(const std::string& material_id);

    /**
     * @brief List all material names in database
     * @return Vector of material names
     */
    std::vector<std::string> list_all_materials() const;

    /**
     * @brief Get database statistics
     * @param total_materials Output: total number of materials
     * @param depth_varying_count Output: number with depth variation
     */
    void get_statistics(int& total_materials, int& depth_varying_count) const;

    /**
     * @brief Create database schema (tables, indices)
     * @throws DatabaseError on database errors
     */
    void create_schema();

    /**
     * @brief Verify database integrity
     * @return true if database is valid
     */
    bool verify_integrity() const;

private:
    sqlite3* db_;                ///< SQLite database handle
    std::string db_path_;        ///< Database file path

    /**
     * @brief Execute SQL statement without results
     * @param sql SQL statement
     * @throws DatabaseError on SQL errors
     */
    void execute(const std::string& sql);

    /**
     * @brief Load material from database (helper)
     * @param material_id UUID to load
     * @return Material properties
     * @throws DatabaseError if not found or database error
     */
    MaterialPropertiesDepth load_material(const std::string& material_id) const;

    /**
     * @brief Generate new UUID v4
     * @return UUID string
     */
    static std::string generate_uuid();
};

} // namespace thermal

#endif // MATERIALS_DB_HPP
