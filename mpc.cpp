#include "ilqr.hpp"
#include "robot_utils.hpp"
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

// Function to load reference trajectory from CSV
std::vector<casadi::DM> loadReferenceFromCSV(const std::string& csv_path, int nq, int max_steps) {
    std::vector<casadi::DM> q_ref_trajectory;
    std::ifstream file(csv_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open reference CSV file: " + csv_path);
    }
    
    std::string line;
    int row_count = 0;
    
    while (std::getline(file, line) && row_count < max_steps) {
        std::stringstream ss(line);
        std::string cell;
        casadi::DM q_ref = casadi::DM::zeros(nq, 1);
        
        int col = 0;
        while (std::getline(ss, cell, ',') && col < nq) {
            q_ref(col) = std::stod(cell);
            col++;
        }
        
        if (col == nq) {
            q_ref_trajectory.push_back(q_ref);
            row_count++;
        } else {
            std::cerr << "Warning: Row " << row_count << " has " << col 
                      << " columns, expected " << nq << std::endl;
        }
    }
    
    file.close();
    
    if (q_ref_trajectory.empty()) {
        throw std::runtime_error("No valid reference data loaded from CSV");
    }
    
    std::cout << "Loaded " << q_ref_trajectory.size() 
              << " reference trajectory points from CSV" << std::endl;
    
    return q_ref_trajectory;
}

// Function to save optimal trajectory to CSV
void saveOptimalTrajectoryToCSV(const std::vector<std::vector<casadi::DM>>& all_optimal_states,
                                const std::vector<double>& time_vec,
                                int nq,
                                const std::string& filename = "q_optimal.csv") {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create output CSV file: " + filename);
    }
    
    // Write header
    file << "time";
    for (int i = 0; i < nq; ++i) {
        file << ",q" << i;
    }
    file << "\n";
    
    // Write data
    for (size_t t = 0; t < all_optimal_states.size(); ++t) {
        file << time_vec[t];
        
        // Extract q from state vector [q, qd]
        for (int i = 0; i < nq; ++i) {
            double q_val = static_cast<double>(all_optimal_states[t][0](i));
            file << "," << q_val;
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Optimal trajectory saved to " << filename << std::endl;
}

int main() {
    try {
        std::cout << "=== Starting Humanoid FOH iLQR MPC ===" << std::endl;
        
        // Robot and MPC Setup
        std::string urdf_path = "/home/prem/mpc/robot/h1_description/urdf/h1.urdf";
        std::string ref_csv_path = "/home/prem/mpc/walk/q_ref.csv";
        
        // Initialize robot
        RobotUtils h1(urdf_path);
        int nq = h1.nq();  // 26 for H1
        int nv = h1.nv();  // 25 for H1
        int nx = h1.nx();  // 51 for H1 (nq + nv)
        int nu = h1.nu();  // 25 for H1
        
        std::cout << "Robot dimensions: nq=" << nq << ", nv=" << nv 
                  << ", nx=" << nx << ", nu=" << nu << std::endl;
        
        // MPC parameters
        double dt = 0.02;       // 50 Hz control rate
        int N = 10;             // Horizon length (0.2 seconds)
        int sim_steps = 500;    // Total simulation time = 10 seconds
        
        // Initialize iLQR solver
        iLQR solver(nx, nu, N, dt, urdf_path);
        solver.setMaxIter(1);           // Real-time constraint
        solver.setTolerance(1e-3);
        solver.setBounds(-100, 100);    // Torque limits ±100 Nm
        solver.setArtificialWeight(1e3);
        solver.setRegularization(1e-2);
        
        // Cost weights (matching paper's approach)
        casadi::DM Q = casadi::DM::eye(nx);
        casadi::DM R = casadi::DM::eye(nu);
        casadi::DM Qf = 2.0 * casadi::DM::eye(nx);  // Terminal weight
        
        // Set different weights for different state components
        for (int i = 0; i < 3; ++i) {
            Q(i, i) = 100.0;  // Base position tracking
        }
        for (int i = 3; i < 7; ++i) {
            Q(i, i) = 50.0;   // Base orientation tracking
        }
        for (int i = 7; i < nq; ++i) {
            Q(i, i) = 10.0;   // Joint position tracking
        }
        for (int i = nq; i < nx; ++i) {
            Q(i, i) = 1.0;    // Velocity tracking
        }
        
        // Control effort weights
        for (int i = 0; i < nu; ++i) {
            R(i, i) = 0.01;   // Low torque penalty
        }
        
        solver.setCostWeights(Q, R, Qf);
        
        // Load reference trajectory from CSV
        std::cout << "Loading reference trajectory..." << std::endl;
        std::vector<casadi::DM> q_ref_data = loadReferenceFromCSV(ref_csv_path, nq, sim_steps + N + 10);
        
        // If reference is shorter than needed, repeat the last state
        while (q_ref_data.size() < sim_steps + N + 1) {
            q_ref_data.push_back(q_ref_data.back());
        }
        
        // Build full state references (q + zero velocities)
        std::vector<casadi::DM> full_x_ref;
        std::vector<casadi::DM> full_u_ref;
        
        for (size_t i = 0; i < q_ref_data.size(); ++i) {
            casadi::DM x_ref = casadi::DM::zeros(nx, 1);
            x_ref(casadi::Slice(0, nq)) = q_ref_data[i];  // q reference
            x_ref(casadi::Slice(nq, nx)) = casadi::DM::zeros(nv, 1);  // qd reference (zero)
            full_x_ref.push_back(x_ref);
            
            if (i < q_ref_data.size() - 1) {
                full_u_ref.push_back(casadi::DM::zeros(nu, 1));  // u reference (zero)
            }
        }
        
        // Initial state (first reference point)
        casadi::DM x0 = full_x_ref[0];
        std::cout << "Initial state set from reference trajectory" << std::endl;
        
        // Storage for simulation results
        casadi::DM xk = x0;
        std::vector<std::vector<casadi::DM>> all_optimal_states;  // Store full trajectories
        std::vector<casadi::DM> all_applied_controls;
        std::vector<double> time_vec;
        std::vector<double> cost_vec;
        
        // Initial trajectory guess
        std::vector<casadi::DM> x_guess(N + 1);
        
        std::cout << "=== Starting MPC Control Loop ===" << std::endl;
        
        // Main MPC Loop
        for (int t_step = 0; t_step < sim_steps; ++t_step) {
            double current_time = t_step * dt;
            time_vec.push_back(current_time);
            
            // Extract reference window (sliding horizon)
            std::vector<casadi::DM> x_ref(N + 1);
            std::vector<casadi::DM> u_ref(N);
            
            for (int k = 0; k <= N; ++k) {
                int ref_idx = std::min(t_step + k, (int)full_x_ref.size() - 1);
                x_ref[k] = full_x_ref[ref_idx];
                if (k < N) {
                    u_ref[k] = full_u_ref[std::min(t_step + k, (int)full_u_ref.size() - 1)];
                }
            }
            
            // Prepare initial guess
            if (t_step == 0) {
                // Initial guess: straight line to reference
                x_guess[0] = xk;
                for (int k = 1; k <= N; ++k) {
                    x_guess[k] = x_ref[k];  // Follow reference
                }
            } else {
                // Warm start: shift previous solution
                x_guess[0] = xk;
                for (int k = 1; k < N; ++k) {
                    x_guess[k] = solver.getStates()[k + 1];
                }
                x_guess[N] = x_ref[N];  // Terminal state from reference
            }
            
            // Solve MPC optimization
            std::cout << "Solving MPC step " << t_step + 1 << "/" << sim_steps << "..." << std::flush;
            
            bool success = solver.solve(x_guess, x_ref, u_ref);
            
            if (!success) {
                std::cerr << "\n iLQR failed at step " << t_step << std::endl;
                break;
            }
            
            // Get optimal trajectory and control
            const auto& optimal_states = solver.getStates();
            const auto& optimal_controls = solver.getControls();
            double optimal_cost = solver.getCost();
            
            // Store optimal trajectory for this time step
            std::vector<casadi::DM> current_optimal_states;
            for (const auto& state : optimal_states) {
                current_optimal_states.push_back(state);
            }
            all_optimal_states.push_back(current_optimal_states);
            
            // Apply first control
            casadi::DM uk = optimal_controls[0];
            all_applied_controls.push_back(uk);
            cost_vec.push_back(optimal_cost);
            
            // Forward simulate (using robot dynamics)
            casadi::DM xdot = solver.humanoidDynamics(xk, uk);
            casadi::DM x_next = xk + dt * xdot;  // Euler integration
            
            // Get contact forces for analysis
            auto contact_forces = solver.getContactForces();
            
            std::cout << " Cost: " << optimal_cost << std::endl;
            
            // Progress monitoring
            if (t_step % 50 == 0) {
                std::cout << "Progress: " << (100.0 * t_step / sim_steps) << "% complete" << std::endl;
                
                // Print key joint positions
                std::cout << "  Base height: " << static_cast<double>(x_next(2)) << " m" << std::endl;
                if (nq > 10) {
                    std::cout << "  Hip joints: [" 
                              << static_cast<double>(x_next(7)) << ", "
                              << static_cast<double>(x_next(8)) << "]" << std::endl;
                }
            }
            
            // Update state for next iteration
            xk = x_next;
        }
        
        std::cout << "\n=== MPC Simulation Complete ===" << std::endl;
        
        // Save optimal trajectories to CSV
        std::cout << "Saving optimal trajectory..." << std::endl;
        saveOptimalTrajectoryToCSV(all_optimal_states, time_vec, nq, "q_optimal.csv");
                
        // Print final statistics
        std::cout << "\n=== Simulation Statistics ===" << std::endl;
        std::cout << "Total time steps: " << sim_steps << std::endl;
        std::cout << "Total time: " << sim_steps * dt << " seconds" << std::endl;
        std::cout << "Average cost: " << std::accumulate(cost_vec.begin(), cost_vec.end(), 0.0) / cost_vec.size() << std::endl;
        std::cout << "Files saved:" << std::endl;
        std::cout << "  - q_optimal.csv (optimal joint trajectories)" << std::endl;
        std::cout << "  - mpc_summary.csv (cost and key metrics)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
}
