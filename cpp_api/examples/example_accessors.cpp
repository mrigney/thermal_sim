/**
 * @file example_accessors.cpp
 * @brief Example usage of free function accessors
 *
 * Demonstrates the getter/setter free functions for MaterialPropertiesDepth.
 * Shows both functional programming style and traditional member access.
 */

#include "materials_db.hpp"
#include "materials_db_accessors.hpp"
#include <iostream>
#include <iomanip>

using namespace thermal;

void print_separator() {
    std::cout << std::string(70, '=') << '\n';
}

void example_basic_setters() {
    std::cout << "\nExample 1: Creating Material Using Free Function Setters\n";
    print_separator();

    MaterialPropertiesDepth mat;

    // Using free function setters
    Name(mat, "Example Material");
    Version(mat, 1);
    Alpha(mat, 0.85);
    Epsilon(mat, 0.90);
    Roughness(mat, 0.005);

    // Set depth-varying properties
    Depths(mat, {0.0, 0.10, 0.25, 0.50});
    K(mat, {0.5, 0.6, 0.7, 0.8});
    Rho(mat, {1600, 1650, 1700, 1750});
    Cp(mat, {850, 850, 850, 850});

    // Set provenance
    SourceDatabase(mat, "Example Database");
    SourceCitation(mat, "Smith et al. (2026)");
    Notes(mat, "Created using accessor functions");

    // Using free function getters
    std::cout << "Material created:\n";
    std::cout << "  Name: " << Name(mat) << '\n';
    std::cout << "  Version: " << Version(mat) << '\n';
    std::cout << "  Alpha: " << Alpha(mat) << '\n';
    std::cout << "  Epsilon: " << Epsilon(mat) << '\n';
    std::cout << "  Roughness: " << Roughness(mat) << " m\n";
    std::cout << "  Source: " << SourceCitation(mat) << '\n';

    // Print thermal properties
    auto depths = Depths(mat);
    auto k_vals = K(mat);

    std::cout << "\n  Thermal properties:\n";
    std::cout << "    Depths: ";
    for (auto d : depths) std::cout << d << " ";
    std::cout << "m\n";

    std::cout << "    k:      ";
    for (auto k : k_vals) std::cout << k << " ";
    std::cout << "W/(m·K)\n";
}

void example_functional_style() {
    std::cout << "\nExample 2: Functional Programming Style\n";
    print_separator();

    MaterialPropertiesDepth mat;

    // Chain-like usage (though C++ doesn't support true chaining without builder pattern)
    Name(mat, "Desert Sand");
    Version(mat, 1);

    // Set all properties in sequence
    Depths(mat, {0.0, 0.05, 0.10, 0.20, 0.50});
    ThermalConductivity(mat, {0.30, 0.35, 0.40, 0.45, 0.50});
    Density(mat, {1500, 1550, 1600, 1650, 1700});
    SpecificHeat(mat, {800, 800, 800, 800, 800});

    Alpha(mat, 0.85);
    Epsilon(mat, 0.90);
    Roughness(mat, 0.001);

    // Query derived properties using functions
    double I_surface = ThermalInertia(mat, 0.0);
    double alpha_thermal = ThermalDiffusivity(mat, 0.0);

    std::cout << "Material: " << Name(mat) << '\n';
    std::cout << "Surface thermal inertia: " << std::fixed << std::setprecision(0)
              << I_surface << " J/(m²·K·s^0.5)\n";
    std::cout << "Surface thermal diffusivity: " << std::scientific << std::setprecision(2)
              << alpha_thermal << " m²/s\n";

    // Interpolation using free functions
    double depth_query = 0.15;  // meters
    double k_at_depth = InterpolateK(mat, depth_query);
    double rho_at_depth = InterpolateRho(mat, depth_query);
    double cp_at_depth = InterpolateCp(mat, depth_query);

    std::cout << "\nInterpolated properties at " << depth_query << " m depth:\n";
    std::cout << "  k:   " << std::fixed << std::setprecision(3) << k_at_depth << " W/(m·K)\n";
    std::cout << "  ρ:   " << std::setprecision(0) << rho_at_depth << " kg/m³\n";
    std::cout << "  cp:  " << cp_at_depth << " J/(kg·K)\n";
}

void example_comparison_styles() {
    std::cout << "\nExample 3: Comparing Access Styles\n";
    print_separator();

    MaterialPropertiesDepth mat;

    std::cout << "Style 1: Direct member access (traditional C++)\n";
    mat.name = "Material A";
    mat.version = 1;
    mat.alpha = 0.85;
    std::cout << "  Name: " << mat.name << ", Alpha: " << mat.alpha << '\n';

    std::cout << "\nStyle 2: Free function access (functional style)\n";
    Name(mat, "Material B");
    Version(mat, 2);
    Alpha(mat, 0.90);
    std::cout << "  Name: " << Name(mat) << ", Alpha: " << Alpha(mat) << '\n';

    std::cout << "\nBoth styles work equally well and can be mixed!\n";
}

void example_with_database() {
    std::cout << "\nExample 4: Using Accessors with Database Operations\n";
    print_separator();

    try {
        // Create material using accessor functions
        MaterialPropertiesDepth new_mat;

        Name(new_mat, "Test Material via Accessors");
        Version(new_mat, 1);

        Depths(new_mat, {0.0, 0.10, 0.25});
        K(new_mat, {0.5, 0.6, 0.7});
        Rho(new_mat, {1600, 1650, 1700});
        Cp(new_mat, {850, 850, 850});

        Alpha(new_mat, 0.80);
        Epsilon(new_mat, 0.92);
        Roughness(new_mat, 0.005);

        SourceDatabase(new_mat, "Accessor Example");
        SourceCitation(new_mat, "Generated programmatically (2026)");
        Notes(new_mat, "Demonstrates accessor function usage");

        // Validate before adding
        Validate(new_mat);
        std::cout << "Material validation: PASSED\n";

        // Add to database
        MaterialDatabase db("test_accessors.db", true);
        std::string mat_id = db.add_material(new_mat);

        std::cout << "Added material to database\n";
        std::cout << "  ID: " << mat_id << '\n';
        std::cout << "  Name: " << Name(new_mat) << '\n';
        std::cout << "  Thermal inertia (surface): "
                  << std::fixed << std::setprecision(0)
                  << ThermalInertia(new_mat, 0.0) << " J/(m²·K·s^0.5)\n";

        // Retrieve and use accessors
        auto retrieved = db.get_material(mat_id);
        if (retrieved) {
            std::cout << "\nRetrieved material from database:\n";
            std::cout << "  Name: " << Name(*retrieved) << '\n';
            std::cout << "  Version: " << Version(*retrieved) << '\n';
            std::cout << "  Source: " << SourceCitation(*retrieved) << '\n';

            // Query properties at different depths
            std::cout << "\nThermal conductivity at different depths:\n";
            for (double depth : {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}) {
                std::cout << "  " << std::setw(6) << std::setprecision(2) << depth
                          << " m:  " << std::setw(6) << std::setprecision(3)
                          << InterpolateK(*retrieved, depth) << " W/(m·K)\n";
            }
        }

        db.close();

    } catch (const DatabaseError& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
}

void example_batch_operations() {
    std::cout << "\nExample 5: Batch Operations with Accessors\n";
    print_separator();

    // Create multiple materials
    std::vector<MaterialPropertiesDepth> materials(3);

    // Material 1
    Name(materials[0], "Sand");
    Alpha(materials[0], 0.85);
    Epsilon(materials[0], 0.90);
    Depths(materials[0], {0.0, 0.5});
    K(materials[0], {0.3, 0.5});
    Rho(materials[0], {1500, 1700});
    Cp(materials[0], {800, 800});

    // Material 2
    Name(materials[1], "Granite");
    Alpha(materials[1], 0.75);
    Epsilon(materials[1], 0.88);
    Depths(materials[1], {0.0, 0.5});
    K(materials[1], {2.5, 2.5});
    Rho(materials[1], {2650, 2650});
    Cp(materials[1], {790, 790});

    // Material 3
    Name(materials[2], "Basalt");
    Alpha(materials[2], 0.80);
    Epsilon(materials[2], 0.95);
    Depths(materials[2], {0.0, 0.5});
    K(materials[2], {1.5, 2.0});
    Rho(materials[2], {2900, 2950});
    Cp(materials[2], {840, 840});

    // Print comparison table
    std::cout << std::setw(15) << "Material"
              << std::setw(10) << "Alpha"
              << std::setw(10) << "Epsilon"
              << std::setw(12) << "k (surf)"
              << std::setw(15) << "I (surf)\n";
    std::cout << std::string(62, '-') << '\n';

    for (const auto& mat : materials) {
        std::cout << std::setw(15) << Name(mat)
                  << std::setw(10) << std::fixed << std::setprecision(2) << Alpha(mat)
                  << std::setw(10) << Epsilon(mat)
                  << std::setw(12) << InterpolateK(mat, 0.0)
                  << std::setw(15) << std::setprecision(0) << ThermalInertia(mat, 0.0)
                  << '\n';
    }
}

int main() {
    std::cout << "C++ Materials Database API - Accessor Functions Examples\n";
    print_separator();

    // Run examples
    example_basic_setters();
    example_functional_style();
    example_comparison_styles();
    example_with_database();
    example_batch_operations();

    std::cout << "\nAll accessor examples completed successfully!\n";
    std::cout << "\nNote: Accessor functions provide flexibility in coding style.\n";
    std::cout << "Use whichever style fits your project best, or mix both!\n";

    return 0;
}
