/**
 * @file materials_db_accessors.hpp
 * @brief Free function accessors for MaterialPropertiesDepth
 *
 * Provides getter/setter free functions for all MaterialPropertiesDepth members.
 * This enables a functional programming style and simplifies property access.
 *
 * @author Thermal Terrain Simulator Project
 * @date January 2026
 * @version 1.0
 */

#ifndef MATERIALS_DB_ACCESSORS_HPP
#define MATERIALS_DB_ACCESSORS_HPP

#include "materials_db.hpp"
#include <string>
#include <vector>

namespace thermal {

// =============================================================================
// String Properties
// =============================================================================

/**
 * @brief Get material ID
 * @param mat Material object
 * @return Material ID (UUID)
 */
inline std::string MaterialId(const MaterialPropertiesDepth& mat) {
    return mat.material_id;
}

/**
 * @brief Set material ID
 * @param mat Material object
 * @param value New material ID
 */
inline void MaterialId(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.material_id = value;
}

/**
 * @brief Get material name
 * @param mat Material object
 * @return Material name
 */
inline std::string Name(const MaterialPropertiesDepth& mat) {
    return mat.name;
}

/**
 * @brief Set material name
 * @param mat Material object
 * @param value New material name
 */
inline void Name(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.name = value;
}

/**
 * @brief Get source database name
 * @param mat Material object
 * @return Source database name
 */
inline std::string SourceDatabase(const MaterialPropertiesDepth& mat) {
    return mat.source_database;
}

/**
 * @brief Set source database name
 * @param mat Material object
 * @param value New source database name
 */
inline void SourceDatabase(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.source_database = value;
}

/**
 * @brief Get source citation
 * @param mat Material object
 * @return Source citation
 */
inline std::string SourceCitation(const MaterialPropertiesDepth& mat) {
    return mat.source_citation;
}

/**
 * @brief Set source citation
 * @param mat Material object
 * @param value New source citation
 */
inline void SourceCitation(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.source_citation = value;
}

/**
 * @brief Get notes
 * @param mat Material object
 * @return Notes
 */
inline std::string Notes(const MaterialPropertiesDepth& mat) {
    return mat.notes;
}

/**
 * @brief Set notes
 * @param mat Material object
 * @param value New notes
 */
inline void Notes(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.notes = value;
}

/**
 * @brief Get supersedes UUID
 * @param mat Material object
 * @return UUID of superseded material
 */
inline std::string Supersedes(const MaterialPropertiesDepth& mat) {
    return mat.supersedes;
}

/**
 * @brief Set supersedes UUID
 * @param mat Material object
 * @param value UUID of material this supersedes
 */
inline void Supersedes(MaterialPropertiesDepth& mat, const std::string& value) {
    mat.supersedes = value;
}

// =============================================================================
// Integer Properties
// =============================================================================

/**
 * @brief Get material version
 * @param mat Material object
 * @return Version number
 */
inline int Version(const MaterialPropertiesDepth& mat) {
    return mat.version;
}

/**
 * @brief Set material version
 * @param mat Material object
 * @param value New version number
 */
inline void Version(MaterialPropertiesDepth& mat, int value) {
    mat.version = value;
}

// =============================================================================
// Scalar Double Properties
// =============================================================================

/**
 * @brief Get solar absorptivity
 * @param mat Material object
 * @return Solar absorptivity [0-1]
 */
inline double Alpha(const MaterialPropertiesDepth& mat) {
    return mat.alpha;
}

/**
 * @brief Set solar absorptivity
 * @param mat Material object
 * @param value Solar absorptivity [0-1]
 */
inline void Alpha(MaterialPropertiesDepth& mat, double value) {
    mat.alpha = value;
}

/**
 * @brief Get thermal emissivity
 * @param mat Material object
 * @return Thermal emissivity [0-1]
 */
inline double Epsilon(const MaterialPropertiesDepth& mat) {
    return mat.epsilon;
}

/**
 * @brief Set thermal emissivity
 * @param mat Material object
 * @param value Thermal emissivity [0-1]
 */
inline void Epsilon(MaterialPropertiesDepth& mat, double value) {
    mat.epsilon = value;
}

/**
 * @brief Get surface roughness
 * @param mat Material object
 * @return Surface roughness [m]
 */
inline double Roughness(const MaterialPropertiesDepth& mat) {
    return mat.roughness;
}

/**
 * @brief Set surface roughness
 * @param mat Material object
 * @param value Surface roughness [m]
 */
inline void Roughness(MaterialPropertiesDepth& mat, double value) {
    mat.roughness = value;
}

// =============================================================================
// Vector Properties
// =============================================================================

/**
 * @brief Get depth values
 * @param mat Material object
 * @return Depth array [m]
 */
inline std::vector<double> Depths(const MaterialPropertiesDepth& mat) {
    return mat.depths;
}

/**
 * @brief Set depth values
 * @param mat Material object
 * @param value New depth array [m]
 */
inline void Depths(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.depths = value;
}

/**
 * @brief Get thermal conductivity array
 * @param mat Material object
 * @return Thermal conductivity array [W/(m·K)]
 */
inline std::vector<double> ThermalConductivity(const MaterialPropertiesDepth& mat) {
    return mat.k;
}

/**
 * @brief Set thermal conductivity array
 * @param mat Material object
 * @param value New thermal conductivity array [W/(m·K)]
 */
inline void ThermalConductivity(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.k = value;
}

/**
 * @brief Get density array
 * @param mat Material object
 * @return Density array [kg/m³]
 */
inline std::vector<double> Density(const MaterialPropertiesDepth& mat) {
    return mat.rho;
}

/**
 * @brief Set density array
 * @param mat Material object
 * @param value New density array [kg/m³]
 */
inline void Density(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.rho = value;
}

/**
 * @brief Get specific heat array
 * @param mat Material object
 * @return Specific heat array [J/(kg·K)]
 */
inline std::vector<double> SpecificHeat(const MaterialPropertiesDepth& mat) {
    return mat.cp;
}

/**
 * @brief Set specific heat array
 * @param mat Material object
 * @param value New specific heat array [J/(kg·K)]
 */
inline void SpecificHeat(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.cp = value;
}

// =============================================================================
// Convenience Aliases (shorter names matching member variables)
// =============================================================================

/**
 * @brief Get thermal conductivity array (short name)
 * @param mat Material object
 * @return Thermal conductivity array [W/(m·K)]
 */
inline std::vector<double> K(const MaterialPropertiesDepth& mat) {
    return mat.k;
}

/**
 * @brief Set thermal conductivity array (short name)
 * @param mat Material object
 * @param value New thermal conductivity array [W/(m·K)]
 */
inline void K(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.k = value;
}

/**
 * @brief Get density array (short name)
 * @param mat Material object
 * @return Density array [kg/m³]
 */
inline std::vector<double> Rho(const MaterialPropertiesDepth& mat) {
    return mat.rho;
}

/**
 * @brief Set density array (short name)
 * @param mat Material object
 * @param value New density array [kg/m³]
 */
inline void Rho(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.rho = value;
}

/**
 * @brief Get specific heat array (short name)
 * @param mat Material object
 * @return Specific heat array [J/(kg·K)]
 */
inline std::vector<double> Cp(const MaterialPropertiesDepth& mat) {
    return mat.cp;
}

/**
 * @brief Set specific heat array (short name)
 * @param mat Material object
 * @param value New specific heat array [J/(kg·K)]
 */
inline void Cp(MaterialPropertiesDepth& mat, const std::vector<double>& value) {
    mat.cp = value;
}

// =============================================================================
// Derived Properties (getters only)
// =============================================================================

/**
 * @brief Compute thermal diffusivity at given depth
 * @param mat Material object
 * @param depth_m Depth in meters (default: 0.0)
 * @return Thermal diffusivity α = k/(ρ·cp) [m²/s]
 */
inline double ThermalDiffusivity(const MaterialPropertiesDepth& mat, double depth_m = 0.0) {
    return mat.thermal_diffusivity(depth_m);
}

/**
 * @brief Compute thermal inertia at given depth
 * @param mat Material object
 * @param depth_m Depth in meters (default: 0.0)
 * @return Thermal inertia I = √(k·ρ·cp) [J/(m²·K·s^0.5)]
 */
inline double ThermalInertia(const MaterialPropertiesDepth& mat, double depth_m = 0.0) {
    return mat.thermal_inertia(depth_m);
}

// =============================================================================
// Interpolation Functions
// =============================================================================

/**
 * @brief Interpolate thermal conductivity to target depth
 * @param mat Material object
 * @param depth_m Target depth [m]
 * @return Interpolated k value [W/(m·K)]
 */
inline double InterpolateK(const MaterialPropertiesDepth& mat, double depth_m) {
    return mat.interpolate_k(depth_m);
}

/**
 * @brief Interpolate density to target depth
 * @param mat Material object
 * @param depth_m Target depth [m]
 * @return Interpolated ρ value [kg/m³]
 */
inline double InterpolateRho(const MaterialPropertiesDepth& mat, double depth_m) {
    return mat.interpolate_rho(depth_m);
}

/**
 * @brief Interpolate specific heat to target depth
 * @param mat Material object
 * @param depth_m Target depth [m]
 * @return Interpolated cp value [J/(kg·K)]
 */
inline double InterpolateCp(const MaterialPropertiesDepth& mat, double depth_m) {
    return mat.interpolate_cp(depth_m);
}

/**
 * @brief Interpolate all properties to target depths
 * @param mat Material object
 * @param target_depths Vector of target depths [m]
 * @param k_out Output thermal conductivity values
 * @param rho_out Output density values
 * @param cp_out Output specific heat values
 */
inline void InterpolateToDepths(
    const MaterialPropertiesDepth& mat,
    const std::vector<double>& target_depths,
    std::vector<double>& k_out,
    std::vector<double>& rho_out,
    std::vector<double>& cp_out
) {
    mat.interpolate_to_depths(target_depths, k_out, rho_out, cp_out);
}

// =============================================================================
// Validation
// =============================================================================

/**
 * @brief Validate material properties
 * @param mat Material object
 * @throws DatabaseError if properties are invalid
 */
inline void Validate(const MaterialPropertiesDepth& mat) {
    mat.validate();
}

} // namespace thermal

#endif // MATERIALS_DB_ACCESSORS_HPP
