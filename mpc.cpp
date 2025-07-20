#include "ilqr.hpp"
#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

int main() {
    std::cout << "Starting FOH iLQR MPC" << std::endl;

    // Simulator and MPC Setup 
    double m = 1.0;         // Pendulum mass
    double L = 1.0;         // Pendulum length
    double dt = 0.05;       // Time step
    int N = 20;             // Horizon length
    int sim_steps = 200;    // Total simulation time = 5 seconds

    // Initial state (hanging down position: pi radians)
    casadi::DM x0 = casadi::DM::vertcat({M_PI/6, 0.0});

    // Initialize solver
    iLQR solver(2, 1, N, dt, m, L);
    solver.setMaxIter(20);
    solver.setTolerance(1e-3);
    solver.setBounds(-50, 50);
    solver.setArtificialWeight(1e3);
    solver.setRegularization(1e-2);
    int nx = 2; int nu = 1;

    casadi::DM Q = 100 * casadi::DM::eye(nx);    // Higher state penalty
    casadi::DM R = 1.0 * casadi::DM::eye(nu);   // Moderate control penalty
    casadi::DM Qf = 1000 * casadi::DM::eye(nx);  // High terminal penalty

    solver.setCostWeights(Q, R, Qf);
    // Sinusoidal reference trajectory generation
    std::vector<casadi::DM> full_x_ref;
    std::vector<casadi::DM> full_u_ref;

    // Sinusoidal trajectory parameters
    double amplitude = M_PI / 4.0;     // 45 degrees amplitude
    double frequency = 0.5;            // 0.5 Hz frequency
    double omega = 2.0 * M_PI * frequency;  // Angular frequency

    for (int i = 0; i <= sim_steps + N; ++i) {
        double t = i * dt;
        // Sinusoidal reference
        double theta_target = amplitude * sin(omega * t);
        double theta_dot_target = amplitude * omega * cos(omega * t);
        full_x_ref.emplace_back(casadi::DM::vertcat({theta_target, theta_dot_target}));
    }

    for (int i = 0; i < sim_steps + N; ++i) {
        full_u_ref.emplace_back(casadi::DM::zeros(1));  
    }

    // Main MPC Loop 
    casadi::DM xk = x0;
    std::vector<casadi::DM> all_x{ x0 };
    std::vector<casadi::DM> all_u;
    std::vector<double> time_vec{ 0.0 };                // Start with t=0
    
    // Store reference trajectories for plotting
    std::vector<double> theta_ref_vec;
    std::vector<double> theta_dot_ref_vec;
    std::vector<casadi::DM> x_guess(N + 1);

    for (int t_step = 0; t_step < sim_steps; ++t_step) {
        // Extract reference window (sliding)
        std::vector<casadi::DM> x_ref(N + 1);
        std::vector<casadi::DM> u_ref(N);

        for (int k = 0; k <= N; ++k) {
            x_ref[k] = full_x_ref[std::min(t_step + k, (int)full_x_ref.size() - 1)];
            if (k < N) u_ref[k] = full_u_ref[std::min(t_step + k, (int)full_u_ref.size() - 1)];
        }

        // Store current reference for plotting
        theta_ref_vec.push_back(static_cast<double>(x_ref[0](0)));
        theta_dot_ref_vec.push_back(static_cast<double>(x_ref[0](1)));
        
        if (t_step == 0) {
            // Initial guess
            x_guess[0] = xk;
            for (int k = 1; k <= N; ++k) {
                // double alpha = static_cast<double>(k) / static_cast<double>(N);
                // x_guess[k] = (1.0 - alpha) * xk + alpha * x_ref[k];
                // double decay = exp(-3.0 * k / N);
                // x_guess[k] = decay * xk + (1.0 - decay) * x_ref[k];
                x_guess[k] = casadi::DM::zeros(nx);  
            }
        } else {
            // Warm start
            x_guess[0] = xk;
            for (int k = 1; k < N; ++k) {
                x_guess[k] = solver.getStates()[k+1];
            }
            x_guess[N] = x_ref[N];
        }

        // Call iLQR solve with trajectory guess
        bool success = solver.solve(x_guess, x_ref, u_ref);
        std::cout << "Solved for step: " << t_step + 1 << " with cost " << solver.getCost() << std::endl;
        
        if (!success) {
            std::cerr << "iLQR failed at step " << t_step << std::endl;
            break;
        }

        // Get control from optimizer
        const auto& u_traj = solver.getControls();
        double uk = static_cast<double>(u_traj[0](0));  // apply first control

        casadi::DM x_next = solver.pendulumDynamics(xk, casadi::DM(uk));
        x_next = xk + dt * x_next;  // Euler integration

        all_u.push_back(casadi::DM(uk));
        all_x.push_back(x_next);
        time_vec.push_back((t_step + 1) * dt);

        xk = x_next;

        if (t_step % 1 == 0) {
            std::cout << "Step " << t_step << ", θ: " << xk(0) << ", u: " << uk << ", ref: " << x_ref[0](0) << std::endl;
        }
    }
    std::cout << "MPC Simulation Complete" << std::endl;
    
    // CSV Export for plotting
    std::cout << "Exporting data to CSV..." << std::endl;
    
    // Extract data for export
    std::vector<double> time_data, theta_data, theta_dot_data, control_data;
    
    for (int i = 0; i < time_vec.size(); ++i) {
        time_data.push_back(time_vec[i]);
        theta_data.push_back(static_cast<double>(all_x[i](0)));
        theta_dot_data.push_back(static_cast<double>(all_x[i](1)));
        
        if (i < all_u.size()) {
            control_data.push_back(static_cast<double>(all_u[i](0)));
        } else {
            control_data.push_back(0.0);
        }
    }
    
    // Export to CSV
    std::ofstream csvFile("mpc_results.csv");
    csvFile << "time,theta_actual,theta_ref,theta_dot_actual,theta_dot_ref,control\n";
    
    for (int i = 0; i < time_data.size(); ++i) {
        double theta_ref = (i < theta_ref_vec.size()) ? theta_ref_vec[i] : theta_ref_vec.back();
        double theta_dot_ref = (i < theta_dot_ref_vec.size()) ? theta_dot_ref_vec[i] : theta_dot_ref_vec.back();
        double control = (i < control_data.size()) ? control_data[i] : 0.0;
        
        csvFile << time_data[i] << ","
                << theta_data[i] << ","
                << theta_ref << ","
                << theta_dot_data[i] << ","
                << theta_dot_ref << ","
                << control << "\n";
    }
    csvFile.close();
    
    return 0;
}
