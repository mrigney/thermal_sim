/**
 * @file example_usage.cpp
 * @brief Example usage of the C++ Materials Database API
 *
 * Demonstrates basic operations: opening database, querying materials,
 * interpolating properties, and adding new materials.
 */

#include "materials_db.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace thermal;

void print_separator() {
    std::cout << std::string(70, '=') << '\n';
}

void example_query_material() {
    std::cout << "\nExample 1: Query Material by Name\n";
    print_separator();

    try {
        // Open existing database
        MaterialDatabase db("../data/materials/materials.db");

        // Query material by name
        auto mat = db.get_material_by_name("Desert Sand (Depth-Varying)");

        if (mat) {
            std::cout << "Material found: " << mat->name << '\n';
            std::cout << "  Version: " << mat->version << '\n';
            std::cout << "  Source: " << mat->source_citation << '\n';
            std::cout << '\n';

            // Print depth-varying properties
            std::cout << "  Depth-varying properties:\n";
            std::cout << "    Depths [m]:  ";
            for (double d : mat->depths) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << d;
            }
            std::cout << '\n';

            std::cout << "    k [W/m/K]:   ";
            for (double k : mat->k) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << k;
            }
            std::cout << '\n';

            std::cout << "    ρ [kg/m³]:   ";
            for (double rho : mat->rho) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(0) << rho;
            }
            std::cout << '\n';

            std::cout << "    cp [J/kg/K]: ";
            for (double cp : mat->cp) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(0) << cp;
            }
            std::cout << '\n';
            std::cout << '\n';

            // Surface properties
            std::cout << "  Surface properties:\n";
            std::cout << "    Solar absorptivity (α): " << mat->alpha << '\n';
            std::cout << "    Thermal emissivity (ε): " << mat->epsilon << '\n';
            std::cout << "    Roughness [m]: " << mat->roughness << '\n';
            std::cout << '\n';

            // Computed properties at surface
            double I_surface = mat->thermal_inertia(0.0);
            double alpha_thermal = mat->thermal_diffusivity(0.0);

            std::cout << "  Computed properties (surface):\n";
            std::cout << "    Thermal inertia: " << std::fixed << std::setprecision(0)
                      << I_surface << " J/(m²·K·s^0.5)\n";
            std::cout << "    Thermal diffusivity: " << std::scientific << std::setprecision(2)
                      << alpha_thermal << " m²/s\n";
        } else {
            std::cout << "Material not found\n";
        }

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

void example_interpolate_properties() {
    std::cout << "\nExample 2: Interpolate Properties to Target Depths\n";
    print_separator();

    try {
        MaterialDatabase db("../data/materials/materials.db");

        auto mat = db.get_material_by_name("Desert Sand (Depth-Varying)");

        if (mat) {
            // Define target depths for solver grid
            std::vector<double> target_depths = {
                0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
            };

            // Interpolate properties
            std::vector<double> k_interp, rho_interp, cp_interp;
            mat->interpolate_to_depths(target_depths, k_interp, rho_interp, cp_interp);

            std::cout << "Interpolated to " << target_depths.size() << " solver depths:\n\n";
            std::cout << std::setw(12) << "Depth [m]"
                      << std::setw(12) << "k [W/m/K]"
                      << std::setw(12) << "ρ [kg/m³]"
                      << std::setw(12) << "cp [J/kg/K]"
                      << std::setw(15) << "I [J/m²/K/s^0.5]\n";
            std::cout << std::string(63, '-') << '\n';

            for (size_t i = 0; i < target_depths.size(); ++i) {
                double I = std::sqrt(k_interp[i] * rho_interp[i] * cp_interp[i]);
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(12) << target_depths[i]
                          << std::setw(12) << k_interp[i]
                          << std::setw(12) << std::setprecision(0) << rho_interp[i]
                          << std::setw(12) << cp_interp[i]
                          << std::setw(15) << std::setprecision(0) << I << '\n';
            }
        }

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

void example_list_all_materials() {
    std::cout << "\nExample 3: List All Materials\n";
    print_separator();

    try {
        MaterialDatabase db("../data/materials/materials.db");

        auto materials = db.list_all_materials();

        std::cout << "Found " << materials.size() << " materials in database:\n\n";

        for (size_t i = 0; i < materials.size(); ++i) {
            std::cout << "  [" << (i+1) << "] " << materials[i] << '\n';
        }

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

void example_database_statistics() {
    std::cout << "\nExample 4: Database Statistics\n";
    print_separator();

    try {
        MaterialDatabase db("../data/materials/materials.db");

        int total, depth_varying;
        db.get_statistics(total, depth_varying);

        std::cout << "Database statistics:\n";
        std::cout << "  Total materials: " << total << '\n';
        std::cout << "  Depth-varying materials: " << depth_varying << '\n';
        std::cout << "  Uniform materials: " << (total - depth_varying) << '\n';
        std::cout << '\n';

        bool ok = db.verify_integrity();
        std::cout << "  Database integrity: " << (ok ? "OK" : "FAILED") << '\n';

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

void example_add_material() {
    std::cout << "\nExample 5: Add New Material\n";
    print_separator();

    try {
        // Open database (or create new one for testing)
        MaterialDatabase db("test_materials.db", true);

        // Create new material
        MaterialPropertiesDepth new_mat;
        new_mat.name = "Test Material";
        new_mat.version = 1;

        // Depth-varying properties (3 depth points)
        new_mat.depths = {0.0, 0.10, 0.25};
        new_mat.k = {0.5, 0.6, 0.7};
        new_mat.rho = {1600, 1650, 1700};
        new_mat.cp = {850, 850, 850};

        // Surface properties
        new_mat.alpha = 0.80;
        new_mat.epsilon = 0.92;
        new_mat.roughness = 0.005;

        // Provenance
        new_mat.source_database = "Test Database";
        new_mat.source_citation = "Example Material (2026)";
        new_mat.notes = "Created for API demonstration";

        // Add to database
        std::string mat_id = db.add_material(new_mat);

        std::cout << "Added material: " << new_mat.name << '\n';
        std::cout << "  Material ID: " << mat_id << '\n';

        // Verify by reading back
        auto retrieved = db.get_material(mat_id);
        if (retrieved) {
            std::cout << "  Verified: Successfully retrieved from database\n";
            std::cout << "  Thermal inertia (surface): "
                      << std::fixed << std::setprecision(0)
                      << retrieved->thermal_inertia(0.0) << " J/(m²·K·s^0.5)\n";
        }

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

int main() {
    std::cout << "C++ Materials Database API - Example Usage\n";
    print_separator();

    // Run examples
    example_query_material();
    example_interpolate_properties();
    example_list_all_materials();
    example_database_statistics();
    example_add_material();

    std::cout << "\nAll examples completed successfully!\n";

    return 0;
}
