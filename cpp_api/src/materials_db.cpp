/**
 * @file materials_db.cpp
 * @brief Implementation of C++ Materials Database API
 */

#include "materials_db.hpp"
#include <algorithm>
#include <sstream>
#include <random>
#include <iomanip>

namespace thermal {

// =============================================================================
// MaterialPropertiesDepth Implementation
// =============================================================================

double MaterialPropertiesDepth::thermal_diffusivity(double depth_m) const {
    double k_val = interpolate_k(depth_m);
    double rho_val = interpolate_rho(depth_m);
    double cp_val = interpolate_cp(depth_m);
    return k_val / (rho_val * cp_val);
}

double MaterialPropertiesDepth::thermal_inertia(double depth_m) const {
    double k_val = interpolate_k(depth_m);
    double rho_val = interpolate_rho(depth_m);
    double cp_val = interpolate_cp(depth_m);
    return std::sqrt(k_val * rho_val * cp_val);
}

double MaterialPropertiesDepth::interpolate_k(double depth_m) const {
    return interp1d(depth_m, depths, k);
}

double MaterialPropertiesDepth::interpolate_rho(double depth_m) const {
    return interp1d(depth_m, depths, rho);
}

double MaterialPropertiesDepth::interpolate_cp(double depth_m) const {
    return interp1d(depth_m, depths, cp);
}

void MaterialPropertiesDepth::interpolate_to_depths(
    const std::vector<double>& target_depths,
    std::vector<double>& k_out,
    std::vector<double>& rho_out,
    std::vector<double>& cp_out
) const {
    const size_t n = target_depths.size();
    k_out.resize(n);
    rho_out.resize(n);
    cp_out.resize(n);

    for (size_t i = 0; i < n; ++i) {
        k_out[i] = interpolate_k(target_depths[i]);
        rho_out[i] = interpolate_rho(target_depths[i]);
        cp_out[i] = interpolate_cp(target_depths[i]);
    }
}

void MaterialPropertiesDepth::validate() const {
    if (name.empty()) {
        throw DatabaseError("Material name cannot be empty");
    }

    if (depths.empty() || k.empty() || rho.empty() || cp.empty()) {
        throw DatabaseError("Material properties cannot be empty");
    }

    if (depths.size() != k.size() || depths.size() != rho.size() ||
        depths.size() != cp.size()) {
        throw DatabaseError("Property arrays must have same length as depths");
    }

    // Check depths are sorted and start at 0
    if (depths[0] != 0.0) {
        throw DatabaseError("Depths must start at 0.0");
    }

    for (size_t i = 1; i < depths.size(); ++i) {
        if (depths[i] <= depths[i-1]) {
            throw DatabaseError("Depths must be strictly increasing");
        }
    }

    // Check physical bounds
    for (double k_val : k) {
        if (k_val <= 0.0) {
            throw DatabaseError("Thermal conductivity must be positive");
        }
    }

    for (double rho_val : rho) {
        if (rho_val <= 0.0) {
            throw DatabaseError("Density must be positive");
        }
    }

    for (double cp_val : cp) {
        if (cp_val <= 0.0) {
            throw DatabaseError("Specific heat must be positive");
        }
    }

    if (alpha < 0.0 || alpha > 1.0) {
        throw DatabaseError("Solar absorptivity must be in [0, 1]");
    }

    if (epsilon < 0.0 || epsilon > 1.0) {
        throw DatabaseError("Thermal emissivity must be in [0, 1]");
    }

    if (roughness < 0.0) {
        throw DatabaseError("Roughness must be non-negative");
    }
}

double MaterialPropertiesDepth::interp1d(
    double x,
    const std::vector<double>& xp,
    const std::vector<double>& fp
) {
    if (xp.size() != fp.size()) {
        throw std::invalid_argument("xp and fp must have same size");
    }

    if (xp.empty()) {
        throw std::invalid_argument("Cannot interpolate empty arrays");
    }

    // Single point - constant
    if (xp.size() == 1) {
        return fp[0];
    }

    // Extrapolate below
    if (x <= xp.front()) {
        return fp.front();
    }

    // Extrapolate above
    if (x >= xp.back()) {
        return fp.back();
    }

    // Binary search for interval
    auto it = std::lower_bound(xp.begin(), xp.end(), x);
    size_t i1 = std::distance(xp.begin(), it);
    size_t i0 = i1 - 1;

    // Linear interpolation
    double x0 = xp[i0];
    double x1 = xp[i1];
    double y0 = fp[i0];
    double y1 = fp[i1];

    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}


// =============================================================================
// MaterialDatabase Implementation
// =============================================================================

MaterialDatabase::MaterialDatabase(const std::string& db_path, bool create_if_missing)
    : db_(nullptr), db_path_(db_path)
{
    int flags = SQLITE_OPEN_READWRITE;
    if (create_if_missing) {
        flags |= SQLITE_OPEN_CREATE;
    }

    int rc = sqlite3_open_v2(db_path.c_str(), &db_, flags, nullptr);

    if (rc != SQLITE_OK) {
        std::string error_msg = sqlite3_errmsg(db_);
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
        throw DatabaseError("Failed to open database '" + db_path + "': " + error_msg);
    }

    // If we created a new database, create the schema
    if (create_if_missing) {
        // Check if tables exist
        const char* check_sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='materials'";
        sqlite3_stmt* stmt;
        rc = sqlite3_prepare_v2(db_, check_sql, -1, &stmt, nullptr);
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            bool table_exists = (rc == SQLITE_ROW);
            sqlite3_finalize(stmt);

            if (!table_exists) {
                create_schema();
            }
        }
    }
}

MaterialDatabase::~MaterialDatabase() {
    close();
}

MaterialDatabase::MaterialDatabase(MaterialDatabase&& other) noexcept
    : db_(other.db_), db_path_(std::move(other.db_path_))
{
    other.db_ = nullptr;
}

MaterialDatabase& MaterialDatabase::operator=(MaterialDatabase&& other) noexcept {
    if (this != &other) {
        close();
        db_ = other.db_;
        db_path_ = std::move(other.db_path_);
        other.db_ = nullptr;
    }
    return *this;
}

void MaterialDatabase::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

void MaterialDatabase::execute(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        std::string error = err_msg ? err_msg : "Unknown error";
        if (err_msg) {
            sqlite3_free(err_msg);
        }
        throw DatabaseError("SQL error: " + error);
    }
}

void MaterialDatabase::create_schema() {
    // Create tables
    const char* schema_sql = R"(
        CREATE TABLE IF NOT EXISTS materials (
            material_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            version INTEGER NOT NULL DEFAULT 1,
            source_database TEXT,
            source_citation TEXT,
            notes TEXT,
            supersedes TEXT,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (supersedes) REFERENCES materials(material_id)
        );

        CREATE TABLE IF NOT EXISTS thermal_properties (
            material_id TEXT NOT NULL,
            depth_m REAL NOT NULL,
            thermal_conductivity REAL NOT NULL,
            density REAL NOT NULL,
            specific_heat REAL NOT NULL,
            PRIMARY KEY (material_id, depth_m),
            FOREIGN KEY (material_id) REFERENCES materials(material_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS radiative_properties (
            material_id TEXT PRIMARY KEY,
            solar_absorptivity REAL NOT NULL,
            thermal_emissivity REAL NOT NULL,
            FOREIGN KEY (material_id) REFERENCES materials(material_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS surface_properties (
            material_id TEXT PRIMARY KEY,
            roughness REAL NOT NULL,
            FOREIGN KEY (material_id) REFERENCES materials(material_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS spectral_emissivity (
            material_id TEXT NOT NULL,
            wavelength_um REAL NOT NULL,
            emissivity REAL NOT NULL,
            PRIMARY KEY (material_id, wavelength_um),
            FOREIGN KEY (material_id) REFERENCES materials(material_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_materials_name ON materials(name);
        CREATE INDEX IF NOT EXISTS idx_thermal_depth ON thermal_properties(material_id, depth_m);
    )";

    execute(schema_sql);
}

std::optional<MaterialPropertiesDepth> MaterialDatabase::get_material(
    const std::string& material_id
) const {
    try {
        return load_material(material_id);
    } catch (const DatabaseError&) {
        return std::nullopt;
    }
}

std::optional<MaterialPropertiesDepth> MaterialDatabase::get_material_by_name(
    const std::string& name
) const {
    // Query material ID by name (case-insensitive)
    const char* sql = "SELECT material_id FROM materials WHERE LOWER(name) = LOWER(?)";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare statement: " + std::string(sqlite3_errmsg(db_)));
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<MaterialPropertiesDepth> result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        result = get_material(id);
    }

    sqlite3_finalize(stmt);
    return result;
}

MaterialPropertiesDepth MaterialDatabase::load_material(const std::string& material_id) const {
    MaterialPropertiesDepth mat;
    mat.material_id = material_id;

    // Load from materials table
    const char* sql1 = "SELECT name, version, source_database, source_citation, notes, supersedes "
                       "FROM materials WHERE material_id = ?";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql1, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare statement");
    }

    sqlite3_bind_text(stmt, 1, material_id.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        throw DatabaseError("Material not found: " + material_id);
    }

    mat.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    mat.version = sqlite3_column_int(stmt, 1);
    if (sqlite3_column_text(stmt, 2)) {
        mat.source_database = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    }
    if (sqlite3_column_text(stmt, 3)) {
        mat.source_citation = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    }
    if (sqlite3_column_text(stmt, 4)) {
        mat.notes = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    }
    if (sqlite3_column_text(stmt, 5)) {
        mat.supersedes = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
    }

    sqlite3_finalize(stmt);

    // Load thermal properties
    const char* sql2 = "SELECT depth_m, thermal_conductivity, density, specific_heat "
                       "FROM thermal_properties WHERE material_id = ? ORDER BY depth_m";

    rc = sqlite3_prepare_v2(db_, sql2, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare thermal properties query");
    }

    sqlite3_bind_text(stmt, 1, material_id.c_str(), -1, SQLITE_TRANSIENT);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        mat.depths.push_back(sqlite3_column_double(stmt, 0));
        mat.k.push_back(sqlite3_column_double(stmt, 1));
        mat.rho.push_back(sqlite3_column_double(stmt, 2));
        mat.cp.push_back(sqlite3_column_double(stmt, 3));
    }

    sqlite3_finalize(stmt);

    // Load radiative properties
    const char* sql3 = "SELECT solar_absorptivity, thermal_emissivity "
                       "FROM radiative_properties WHERE material_id = ?";

    rc = sqlite3_prepare_v2(db_, sql3, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare radiative properties query");
    }

    sqlite3_bind_text(stmt, 1, material_id.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        mat.alpha = sqlite3_column_double(stmt, 0);
        mat.epsilon = sqlite3_column_double(stmt, 1);
    }

    sqlite3_finalize(stmt);

    // Load surface properties
    const char* sql4 = "SELECT roughness FROM surface_properties WHERE material_id = ?";

    rc = sqlite3_prepare_v2(db_, sql4, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare surface properties query");
    }

    sqlite3_bind_text(stmt, 1, material_id.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        mat.roughness = sqlite3_column_double(stmt, 0);
    }

    sqlite3_finalize(stmt);

    return mat;
}

std::string MaterialDatabase::add_material(const MaterialPropertiesDepth& material) {
    // Validate first
    material.validate();

    // Generate UUID if not provided
    std::string mat_id = material.material_id;
    if (mat_id.empty()) {
        mat_id = generate_uuid();
    }

    // Begin transaction
    execute("BEGIN TRANSACTION");

    try {
        // Insert into materials table
        const char* sql1 = "INSERT INTO materials (material_id, name, version, source_database, "
                          "source_citation, notes, supersedes) VALUES (?, ?, ?, ?, ?, ?, ?)";

        sqlite3_stmt* stmt;
        int rc = sqlite3_prepare_v2(db_, sql1, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw DatabaseError("Failed to prepare insert statement");
        }

        sqlite3_bind_text(stmt, 1, mat_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, material.name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, material.version);
        sqlite3_bind_text(stmt, 4, material.source_database.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, material.source_citation.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, material.notes.c_str(), -1, SQLITE_TRANSIENT);
        if (!material.supersedes.empty()) {
            sqlite3_bind_text(stmt, 7, material.supersedes.c_str(), -1, SQLITE_TRANSIENT);
        } else {
            sqlite3_bind_null(stmt, 7);
        }

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            throw DatabaseError("Failed to insert material");
        }
        sqlite3_finalize(stmt);

        // Insert thermal properties
        const char* sql2 = "INSERT INTO thermal_properties (material_id, depth_m, "
                          "thermal_conductivity, density, specific_heat) VALUES (?, ?, ?, ?, ?)";

        for (size_t i = 0; i < material.depths.size(); ++i) {
            rc = sqlite3_prepare_v2(db_, sql2, -1, &stmt, nullptr);
            if (rc != SQLITE_OK) {
                throw DatabaseError("Failed to prepare thermal insert");
            }

            sqlite3_bind_text(stmt, 1, mat_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(stmt, 2, material.depths[i]);
            sqlite3_bind_double(stmt, 3, material.k[i]);
            sqlite3_bind_double(stmt, 4, material.rho[i]);
            sqlite3_bind_double(stmt, 5, material.cp[i]);

            if (sqlite3_step(stmt) != SQLITE_DONE) {
                sqlite3_finalize(stmt);
                throw DatabaseError("Failed to insert thermal properties");
            }
            sqlite3_finalize(stmt);
        }

        // Insert radiative properties
        const char* sql3 = "INSERT INTO radiative_properties (material_id, solar_absorptivity, "
                          "thermal_emissivity) VALUES (?, ?, ?)";

        rc = sqlite3_prepare_v2(db_, sql3, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw DatabaseError("Failed to prepare radiative insert");
        }

        sqlite3_bind_text(stmt, 1, mat_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 2, material.alpha);
        sqlite3_bind_double(stmt, 3, material.epsilon);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            throw DatabaseError("Failed to insert radiative properties");
        }
        sqlite3_finalize(stmt);

        // Insert surface properties
        const char* sql4 = "INSERT INTO surface_properties (material_id, roughness) VALUES (?, ?)";

        rc = sqlite3_prepare_v2(db_, sql4, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw DatabaseError("Failed to prepare surface insert");
        }

        sqlite3_bind_text(stmt, 1, mat_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 2, material.roughness);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            throw DatabaseError("Failed to insert surface properties");
        }
        sqlite3_finalize(stmt);

        // Commit transaction
        execute("COMMIT");

    } catch (...) {
        execute("ROLLBACK");
        throw;
    }

    return mat_id;
}

void MaterialDatabase::update_material(const MaterialPropertiesDepth& material) {
    if (material.material_id.empty()) {
        throw DatabaseError("Cannot update material without material_id");
    }

    // Validate
    material.validate();

    // Delete and re-insert (simpler than selective update)
    delete_material(material.material_id);
    add_material(material);
}

void MaterialDatabase::delete_material(const std::string& material_id) {
    const char* sql = "DELETE FROM materials WHERE material_id = ?";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare delete statement");
    }

    sqlite3_bind_text(stmt, 1, material_id.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw DatabaseError("Failed to delete material");
    }

    sqlite3_finalize(stmt);
}

std::vector<std::string> MaterialDatabase::list_all_materials() const {
    std::vector<std::string> materials;

    const char* sql = "SELECT name FROM materials ORDER BY name";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw DatabaseError("Failed to prepare list query");
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        materials.push_back(name);
    }

    sqlite3_finalize(stmt);
    return materials;
}

void MaterialDatabase::get_statistics(int& total_materials, int& depth_varying_count) const {
    // Total materials
    const char* sql1 = "SELECT COUNT(*) FROM materials";

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db_, sql1, -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        total_materials = sqlite3_column_int(stmt, 0);
    } else {
        total_materials = 0;
    }
    sqlite3_finalize(stmt);

    // Depth-varying materials (more than 1 depth point)
    const char* sql2 = "SELECT COUNT(DISTINCT material_id) FROM thermal_properties "
                      "GROUP BY material_id HAVING COUNT(*) > 1";

    sqlite3_prepare_v2(db_, sql2, -1, &stmt, nullptr);
    depth_varying_count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        depth_varying_count++;
    }
    sqlite3_finalize(stmt);
}

bool MaterialDatabase::verify_integrity() const {
    const char* sql = "PRAGMA integrity_check";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }

    bool ok = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        ok = (std::string(result) == "ok");
    }

    sqlite3_finalize(stmt);
    return ok;
}

std::string MaterialDatabase::generate_uuid() {
    // Simple UUID v4 generator
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');

    // UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    oss << std::setw(8) << dis(gen) << '-';
    oss << std::setw(4) << (dis(gen) & 0xFFFF) << '-';
    oss << std::setw(4) << ((dis(gen) & 0x0FFF) | 0x4000) << '-';
    oss << std::setw(4) << ((dis(gen) & 0x3FFF) | 0x8000) << '-';
    oss << std::setw(8) << dis(gen);
    oss << std::setw(4) << (dis(gen) & 0xFFFF);

    return oss.str();
}

} // namespace thermal
