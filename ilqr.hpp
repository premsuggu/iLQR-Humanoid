#pragma once
#include <vector>
#include <casadi/casadi.hpp>

class iLQR {
public:
    // Constructor & initialization
    iLQR(int state_dim, int control_dim, int horizon, double dt, double m, double L);
    
    // Main solve function
    bool solve(const std::vector<casadi::DM>& x_guess, const std::vector<casadi::DM>& x_ref, const std::vector<casadi::DM>& u_ref);
    
    // Getters
    std::vector<casadi::DM> getStates() const { return x_traj_; }
    std::vector<casadi::DM> getControls() const { return u_traj_; }
    double getCost() const { return current_cost_; }
    casadi::DM pendulumDynamics(const casadi::DM& x, const casadi::DM& u);
    
    // Set Parameters
    void setMaxIter(int max_iter) { max_iter_ = max_iter; }
    void setCostWeights(const casadi::DM& Q, const casadi::DM& R, const casadi::DM& Qf) {Q_ = Q; R_ = R; Qf_ = Qf;};
    void setTolerance(double tol) { tolerance_ = tol; }
    void setRegularization(double reg) { reg_ = reg; }
    void setArtificialWeight(double weight) { rho_art_ = weight; }
    void setBounds(double u_min, double u_max) {u_min_ = u_min; u_max_ = u_max;}

private:
    // Problem dimensions
    int nx_, nu_, N_;
    double dt_;

    // Pendulum parameters
    double m_; double L_; double g_;
    
    // Problem functions 
    casadi::DM Q_, R_, Qf_;

    // Symbolic variables
    std::vector<casadi::SX> x_sym_, u_sym_;             // State & Controlsymbols
    std::vector<casadi::SX> x_mid_sym_, u_mid_sym_;     // Midpoint state & controlsymbols
    std::vector<casadi::SX> u_art_sym_;                 // Artificial control symbols
    std::vector<casadi::SX> x_ref_sym_, u_ref_sym_;     // Reference symbols
    std::vector<casadi::SX> mu_sym_;
    std::vector<casadi::SX> lambda_sym_;

    // Trajectories
    std::vector<casadi::DM> x_traj_, u_traj_;      // states & controls
    std::vector<casadi::DM> x_mid_, u_mid_;        // midpoint states & controls(FOH)
    std::vector<casadi::DM> u_art_;                // artificial controls
    double u_min_, u_max_;                         // bounds

    // Reference/Desired trajecotry
    std::vector<casadi::DM> x_ref_, u_ref_;

    // Constraints (Augmented Lagrangian)
    std::vector<casadi::DM> lambda_;      // multipliers
    std::vector<casadi::DM> mu_;          // penalty params

    // Dynamics & Derivatives
    casadi::Function dynamics_func_;
    casadi::Function foh_dynamics_func_;
    casadi::Function dynamics_jac_A_, dynamics_jac_B_, dynamics_jac_C_;

    // Cost & Derivatives
    casadi::Function stage_cost_func_;
    casadi::Function constraint_cost_func_;
    casadi::Function total_cost_func_;

    std::vector<casadi::Function> cost_grad_x_, cost_grad_u_, cost_grad_y_, cost_grad_v_;
    std::vector<casadi::Function> cost_hess_xx_, cost_hess_uu_, cost_hess_yy_, cost_hess_vv_;
    std::vector<casadi::Function> cost_hess_xu_, cost_hess_xy_, cost_hess_xv_;
    std::vector<casadi::Function> cost_hess_ux_, cost_hess_uy_, cost_hess_uv_;
    std::vector<casadi::Function> cost_hess_yx_, cost_hess_yu_, cost_hess_yv_;
    std::vector<casadi::Function> cost_hess_vx_, cost_hess_vu_, cost_hess_vy_;

    // Constraints
    std::vector<casadi::Function> constraint_grad_u_, constraint_grad_x_;
    std::vector<casadi::Function> constraint_hess_ux_, constraint_hess_xx_, constraint_hess_uu_;

    // Derivatives
    std::vector<casadi::DM> S_, s_;           // cost-to-go Hessian/gradient
    std::vector<casadi::DM> A_, B_, C_;       // dynamics jacobians

    std::vector<casadi::DM> L_x_, L_u_, L_y_, L_v_;           // First derivatives
    std::vector<casadi::DM> L_xx_, L_uu_, L_yy_, L_vv_;       // Second derivatives (diagonal)
    std::vector<casadi::DM> L_xu_, L_xy_, L_xv_;              // Cross derivatives with x
    std::vector<casadi::DM> L_ux_, L_uy_, L_uv_;              // Cross derivatives with u
    std::vector<casadi::DM> L_yx_, L_yu_, L_yv_;              // Cross derivatives with y
    std::vector<casadi::DM> L_vx_, L_vu_, L_vy_;              // Cross derivatives with v

    // Constraints
    std::vector<casadi::DM> C_u_, C_x_, C_v_;
    std::vector<casadi::DM> C_uu_, C_xx_, C_ux_;
    
    // Q-function
    std::vector<casadi::DM> Q_x_, Q_u_, Q_v_;   
    std::vector<casadi::DM> Q_xx_, Q_uu_, Q_vv_;   
    std::vector<casadi::DM> Q_ux_, Q_vx_, Q_uv_;

    // Gains (FOH)
    std::vector<casadi::DM> K_;           // feedback gains    ->    K = -Q_{vv}^{-1} Q_{vx}
    std::vector<casadi::DM> b_;           // feedthrough gains (FOH)    ->    b = -Q_{vv}^{-1} Q_{vu}
    std::vector<casadi::DM> d_;           // feedforward gains    ->     d = -Q_{vv}^{-1} Q_v 
    
    // Algorithm parameters
    int max_iter_ = 100;
    double tolerance_ = 1e-6;
    double reg_ = 1e-6;
    double rho_art_ = 1e3;
    std::vector<double> alpha_schedule_;
    
    // Cost tracking
    double current_cost_;
    double previous_cost_;
    bool converged_;

    // Functions
    void createSymbolicFramework();
    // helper functions
    casadi::SX createStageCost();
    casadi::SX createConstraintCost();
    casadi::SX createFOHDynamics();
    casadi::SX pendulumContinuousDynamics(const casadi::SX& x, const casadi::SX& u);
    void createDynamicsJacobians(const casadi::SX& foh_dynamics);
    void collectAllVariables(std::vector<casadi::SX>& all_vars);
    void createCostDerivativeFunctions(const casadi::SX& stage_cost, const std::vector<casadi::SX>& all_vars);
    void createConstraintDerivativeFunctions(const casadi::SX& constraint_cost, const std::vector<casadi::SX>& all_vars);

    // TO update Cost derivatives
    void quadratizeCost();
    void linearizeDynamics();
    void linearizeConstraints(const std::vector<casadi::DM>& all_inputs);
    void collectAllInputs(std::vector<casadi::DM>& all_inputs);    // helper

    void backwardPass();
    // helpers
    void computeQderivatives(int k);
    void computeGains(int k);
    void updateValueFunction(int k);

    bool forwardPass();
    // Helpers for forward pass
    double simulateTrajectory(double alpha);
    double evaluateTrajectory(const std::vector<casadi::DM>& x_traj,
                               const std::vector<casadi::DM>& u_traj,
                               const std::vector<casadi::DM>& u_art);
    
    // Solver Helpers
    void initializeTrajectory(const std::vector<casadi::DM>& x_guess);
    
    // Utility Functions
    void updateMultipliers();
    double computeCost();
    bool checkConvergence();

    // Regularization
    void applyRegularization(int k);
};
