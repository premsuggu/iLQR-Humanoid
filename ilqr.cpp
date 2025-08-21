#include "ilqr.hpp"

iLQR::iLQR(int state_dim, int control_dim, int horizon, double dt, const std::string& urdf_path) 
    : nx_(state_dim), nu_(control_dim), N_(horizon), dt_(dt){
    
    robot_utils_ = std::make_unique<RobotUtils>(urdf_path);
    
    // Verify dimensions match
    if (robot_utils_->nx() != state_dim || robot_utils_->nu() != control_dim) {
        throw std::runtime_error("Dimension mismatch between robot model and iLQR settings");
    }
    
    // PDF Reference: Section 8.3 - FOH requires midpoint variables
    x_traj_.resize(N_ + 1);          // States: x_0, x_1, ..., x_N
    u_traj_.resize(N_);              // Controls: u_0, u_1, ..., u_{N-1}
    u_art_.resize(N_);               // Artificial controls for infeasible trajectories
    
    // Reference trajectories
    x_ref_.resize(N_ + 1);
    u_ref_.resize(N_);
    
    //  SYMBOLIC VARIABLES
    x_sym_.resize(N_ + 1);           // Symbolic states
    u_sym_.resize(N_);               // Symbolic controls
    u_art_sym_.resize(N_);           // Symbolic artificial controls
    x_ref_sym_.resize(N_ + 1);       // Symbolic reference states
    u_ref_sym_.resize(N_);           // Symbolic reference controls
    
    // ============ CONSTRAINT HANDLING ============
    // PDF Reference: Section 3.1 - Augmented Lagrangian
    lambda_.resize(N_);              // Lagrange multipliers
    mu_.resize(N_);                  // Penalty parameters
    lambda_sym_.resize(N_);          // Symbolic multipliers
    mu_sym_.resize(N_);              // Symbolic penalty parameters
    
    // ============ FOH GAINS ============
    // PDF Reference: Section 8.6 - FOH requires three types of gains
    K_.resize(N_);                   // Feedback gains: K_k
    b_.resize(N_);                   // Feedthrough gains: b_k (FOH-specific)
    d_.resize(N_);                   // Feedforward gains: d_k
    
    // ============ VALUE FUNCTION ============
    // PDF Reference: Section 8.4 - Control-dependent value function
    S_.resize(N_ + 1);               // Value function Hessian: (n+m) x (n+m)
    s_.resize(N_ + 1);               // Value function gradient: (n+m) x 1
    
    // ============ DYNAMICS JACOBIANS ============
    // PDF Reference: Section 8.3, Equation (43)
    A_.resize(N_);                   // ∂f_d/∂x
    B_.resize(N_);                   // ∂f_d/∂u_k
    C_.resize(N_);                   // ∂f_d/∂u_{k+1} (FOH-specific)
    
    // ============ STAGE COST DERIVATIVES ============
    // PDF Reference: Section 8.7.2 - All required derivatives
    
    // First derivatives
    L_x_.resize(N_ + 1);             // ∂L/∂x_k for k = 0...N
    L_u_.resize(N_);                 // ∂L/∂u_k for k = 0...N-1
    L_y_.resize(N_);                 // ∂L/∂y_k (y = x_{k+1}) for k = 0...N-1
    L_v_.resize(N_ - 1);             // ∂L/∂v_k (v = u_{k+1}) for k = 0...N-2
    
    // Second derivatives (diagonal)
    L_xx_.resize(N_ + 1);            // ∂²L/∂x_k²
    L_uu_.resize(N_);                // ∂²L/∂u_k²
    L_yy_.resize(N_);                // ∂²L/∂y_k²
    L_vv_.resize(N_ - 1);            // ∂²L/∂v_k²
    
    // Cross derivatives
    L_xu_.resize(N_);                // ∂²L/∂x_k∂u_k
    L_xy_.resize(N_);                // ∂²L/∂x_k∂y_k
    L_xv_.resize(N_ - 1);            // ∂²L/∂x_k∂v_k
    L_uy_.resize(N_);                // ∂²L/∂u_k∂y_k
    L_uv_.resize(N_ - 1);            // ∂²L/∂u_k∂v_k
    L_yv_.resize(N_ - 1);            // ∂²L/∂y_k∂v_k
    
    // Symmetric cross derivatives (for completeness)
    L_ux_.resize(N_);                // ∂²L/∂u_k∂x_k = L_xu^T
    L_yx_.resize(N_);                // ∂²L/∂y_k∂x_k = L_xy^T
    L_vx_.resize(N_ - 1);            // ∂²L/∂v_k∂x_k = L_xv^T
    L_yu_.resize(N_);                // ∂²L/∂y_k∂u_k = L_uy^T
    L_vu_.resize(N_ - 1);            // ∂²L/∂v_k∂u_k = L_uv^T
    L_vy_.resize(N_ - 1);            // ∂²L/∂v_k∂y_k = L_yv^T
    
    // ============ SYMBOLIC DERIVATIVE FUNCTIONS ============
    // These will store CasADi functions for automatic differentiation
    cost_grad_x_.resize(N_ + 1);
    cost_grad_u_.resize(N_);
    cost_grad_y_.resize(N_);
    cost_grad_v_.resize(N_ - 1);
    
    cost_hess_xx_.resize(N_ + 1);
    cost_hess_uu_.resize(N_);
    cost_hess_yy_.resize(N_);
    cost_hess_vv_.resize(N_ - 1);
    
    cost_hess_xu_.resize(N_);
    cost_hess_xy_.resize(N_);
    cost_hess_xv_.resize(N_ - 1);
    cost_hess_uy_.resize(N_);
    cost_hess_uv_.resize(N_ - 1);
    cost_hess_yv_.resize(N_ - 1);
    
    cost_hess_ux_.resize(N_);
    cost_hess_yx_.resize(N_);
    cost_hess_vx_.resize(N_ - 1);
    cost_hess_yu_.resize(N_);
    cost_hess_vu_.resize(N_ - 1);
    cost_hess_vy_.resize(N_ - 1);

    constraint_grad_u_.resize(N_);  constraint_hess_ux_.resize(N_);  constraint_hess_uu_.resize(N_);
    constraint_grad_x_.resize(N_);  constraint_hess_xx_.resize(N_);

    // Numerical constraint derivatives
    C_u_.resize(N_);        C_uu_.resize(N_);
    C_x_.resize(N_);        C_xx_.resize(N_);
    C_ux_.resize(N_);

    // ============ Q-FUNCTION DERIVATIVES ============
    // PDF Reference: Section 8.7.3 - Q = G + H formulation
    Q_x_.resize(N_);        Q_u_.resize(N_);       Q_v_.resize(N_ - 1);         
    Q_xx_.resize(N_);       Q_uu_.resize(N_);      Q_vv_.resize(N_ - 1);       
    Q_ux_.resize(N_);       Q_vx_.resize(N_ - 1);  Q_uv_.resize(N_ - 1); 
        
    // ============ INITIALIZE NUMERICAL VALUES ============
    for (int k = 0; k <= N_; k++) {
        if (k < N_) {
            x_traj_[k] = casadi::DM::zeros(nx_);
            u_traj_[k] = casadi::DM::zeros(nu_);
            u_art_[k] = casadi::DM::zeros(nx_);
            
            // Initialize gains
            K_[k] = casadi::DM::zeros(nu_, nx_);
            b_[k] = casadi::DM::zeros(nu_, nu_);
            d_[k] = casadi::DM::zeros(nu_);
            
            // Initialize constraint multipliers
            lambda_[k] = casadi::DM::zeros(2);      // Two constraints: upper and lower bounds
            mu_[k] = casadi::DM::ones(2);           // Initialize penalty parameters to 1.0
            
        } else {
            x_traj_[k] = casadi::DM::zeros(nx_);
        }
        
        // Initialize value function with proper (n+m) x (n+m) structure
        S_[k] = casadi::DM::zeros(nx_ + nu_, nx_ + nu_);
        s_[k] = casadi::DM::zeros(nx_ + nu_);
    }
    
    // ============ REFERENCE TRAJECTORY INITIALIZATION ============
    for (int k = 0; k <= N_; k++) {
        x_ref_[k] = casadi::DM::zeros(nx_);
        if (k < N_) {
            u_ref_[k] = casadi::DM::zeros(nu_);
        }
    }
    
    // ============ COST TRACKING ============
    current_cost_ = 1e5;
    previous_cost_ = 1e5;
    converged_ = false;
    
    // ============ COST WEIGHTS ============
    Q_ = 1 * casadi::DM::eye(nx_);       // State cost weight
    R_ = 0.01 * casadi::DM::eye(nu_);       // Control cost weight
    Qf_ = 10 * casadi::DM::eye(nx_);      // Terminal cost weight
    
    // ============ ALGORITHM PARAMETERS ============
    max_iter_ = 1;
    tolerance_ = 1e-6;
    reg_ = 1e-6;
    rho_art_ = 1e3;                  // Artificial control penalty
    alpha_schedule_ = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625};
    
    // CONSTRAINT BOUNDS ============
    u_min_ = -100;
    u_max_ = 100;

    // ============ INITIALIZE SYMBOLIC FRAMEWORK ============
    createSymbolicFramework();
    std::cout << "Symbolic framework created successfully!" << std::endl;
}

// PDF Reference: Section 8.3 "Preliminaries" and Section 8.7 "Calculating L"
void iLQR::createSymbolicFramework() {

    for (int k = 0; k <= N_; k++) {
        x_sym_[k] = casadi::SX::sym("x_" + std::to_string(k), nx_);
        x_ref_sym_[k] = casadi::SX::sym("x_ref_" + std::to_string(k), nx_);
        
        if (k < N_) {
            u_sym_[k] = casadi::SX::sym("u_" + std::to_string(k), nu_);
            u_ref_sym_[k] = casadi::SX::sym("u_ref_" + std::to_string(k), nu_);
            u_art_sym_[k] = casadi::SX::sym("u_art_" + std::to_string(k), nx_);
        }
    }
    
    // Constraint multipliers - PDF Reference: Section 3.1
    for (int k = 0; k < N_; k++) {
        lambda_sym_[k] = casadi::SX::sym("lambda_" + std::to_string(k), 2);
        mu_sym_[k] = casadi::SX::sym("mu_" + std::to_string(k), 2);
    }
    
    // Create cost functions
    std::cout << "Creating cost cost..." << std::endl;
    casadi::SX stage_cost = createStageCost();
    std::cout << "Creating constraint cost..." << std::endl;
    casadi::SX constraint_cost = createConstraintCost();
    casadi::SX total_cost = stage_cost + constraint_cost;
    
    std::vector<casadi::SX> all_vars;
    collectAllVariables(all_vars);
    
    total_cost_func_ = casadi::Function("total_cost", all_vars, {total_cost});
    stage_cost_func_ = casadi::Function("stage_cost", all_vars, {stage_cost});
    constraint_cost_func_ = casadi::Function("constraint_cost", all_vars, {constraint_cost});
    
    // FOH dynamics - PDF Reference: Section 8.3, Equation (43)
    std::cout << "Creating FOH dynamics..." << std::endl;
    casadi::SX foh_dynamics = createFOHDynamics();
    foh_dynamics_func_ = casadi::Function("foh_dynamics", {x_sym_[0], u_sym_[0], u_sym_[1]}, {foh_dynamics});
    
    createDynamicsJacobians(foh_dynamics);
    std::cout << "computing derivatives" << std::endl;
    // Create all derivative functions
    createCostDerivativeFunctions(stage_cost, all_vars);
    createConstraintDerivativeFunctions(constraint_cost, all_vars);  
}

// Create stage cost with Simpson's integration - PDF Reference: Section 8.4.1
casadi::SX iLQR::createStageCost() {
    casadi::SX stage_cost = 0;
    
    // Terminal cost using robot_utils
    casadi::SX terminal_cost = robot_utils_->symCostStage(x_sym_[N_], casadi::SX::zeros(nu_, 1), x_ref_sym_[N_], casadi::SX::zeros(nu_, 1));
    stage_cost += robot_utils_->weights.terminal_scale * terminal_cost;

    // Simpson's integration for stage costs
    for (int k = 0; k < N_; k++) {
        
        // Left endpoint cost using robot_utils
        casadi::SX cost_left = robot_utils_->symCostStage(x_sym_[k], u_sym_[k], x_ref_sym_[k], u_ref_sym_[k]);
        
        // Right endpoint cost
        casadi::SX cost_right = robot_utils_->symCostStage(x_sym_[k+1], 
                                                          (k < N_-1) ? u_sym_[k+1] : casadi::SX::zeros(nu_, 1),
                                                          x_ref_sym_[k+1], 
                                                          (k < N_-1) ? u_ref_sym_[k+1] : casadi::SX::zeros(nu_, 1));
        
        // Midpoint cost (if k < N-1)
        casadi::SX cost_mid = 0;
        if (k < N_ - 1) {
            // Compute midpoint dynamics using robot_utils
            casadi::SX f_k = robot_utils_->symFloatingBaseDynamics(x_sym_[k], u_sym_[k]);
            casadi::SX f_k1 = robot_utils_->symFloatingBaseDynamics(x_sym_[k+1], u_sym_[k+1]);

            // Hermite-Simpson midpoint state
            casadi::SX x_mid_expr = 0.5 * (x_sym_[k] + x_sym_[k+1]) + (dt_/8.0) * (f_k - f_k1);
            casadi::SX u_mid_expr = 0.5 * (u_sym_[k] + u_sym_[k+1]);
            
            // Midpoint references
            casadi::SX x_mid_ref = 0.5 * (x_ref_sym_[k] + x_ref_sym_[k+1]);
            casadi::SX u_mid_ref = 0.5 * (u_ref_sym_[k] + u_ref_sym_[k+1]);
            
            // Midpoint cost using robot_utils
            cost_mid = robot_utils_->symCostStage(x_mid_expr, u_mid_expr,  x_mid_ref, u_mid_ref);
        }
        
        // Simpson's rule
        stage_cost += (dt_ / 6.0) * (cost_left + 4 * cost_mid + cost_right);
        
        // Artificial control penalty
        stage_cost += 0.5 * rho_art_ * casadi::SX::dot(u_art_sym_[k], u_art_sym_[k]);
    }
    
    return stage_cost;
}


// Create constraint cost - PDF Reference: Section 3.1
casadi::SX iLQR::createConstraintCost() {
    casadi::SX constraint_cost = 0;
    for (int k = 0; k < N_; k++) {
        // Use robot_utils constraint penalty function
        constraint_cost += robot_utils_->symConstraintPenalty(x_sym_[k], u_sym_[k]);
    }
    
    return constraint_cost;
}

// Create FOH dynamics - PDF Reference: Section 8.3
casadi::SX iLQR::createFOHDynamics() {
    // FOH control averaging
    casadi::SX u_avg = 0.5 * (u_sym_[0] + u_sym_[1]);
    casadi::SX f_continuous = robot_utils_->symFloatingBaseDynamics(x_sym_[0], u_avg);              // Get continuous dynamics
    return x_sym_[0] + dt_ * f_continuous;          // Euler integration
}

// Create dynamics Jacobians - PDF Reference: Section 8.3, Equation (43)
void iLQR::createDynamicsJacobians(const casadi::SX& foh_dynamics) {
    casadi::SX A_expr = casadi::SX::jacobian(foh_dynamics, x_sym_[0]);
    casadi::SX B_expr = casadi::SX::jacobian(foh_dynamics, u_sym_[0]);
    casadi::SX C_expr = casadi::SX::jacobian(foh_dynamics, u_sym_[1]);
    
    dynamics_jac_A_ = casadi::Function("dyn_jac_A", {x_sym_[0], u_sym_[0], u_sym_[1]}, {A_expr});
    dynamics_jac_B_ = casadi::Function("dyn_jac_B", {x_sym_[0], u_sym_[0], u_sym_[1]}, {B_expr});
    dynamics_jac_C_ = casadi::Function("dyn_jac_C", {x_sym_[0], u_sym_[0], u_sym_[1]}, {C_expr});
    std::cout << "computed the dynamics derivatives" << std::endl;
}

// Collect all variables in correct order
void iLQR::collectAllVariables(std::vector<casadi::SX>& all_vars) {
    all_vars.clear();
    all_vars.insert(all_vars.end(), x_sym_.begin(), x_sym_.end());
    all_vars.insert(all_vars.end(), x_ref_sym_.begin(), x_ref_sym_.end());
    all_vars.insert(all_vars.end(), u_sym_.begin(), u_sym_.end());
    all_vars.insert(all_vars.end(), u_ref_sym_.begin(), u_ref_sym_.end());
    all_vars.insert(all_vars.end(), u_art_sym_.begin(), u_art_sym_.end());
    all_vars.insert(all_vars.end(), lambda_sym_.begin(), lambda_sym_.end());
    all_vars.insert(all_vars.end(), mu_sym_.begin(), mu_sym_.end());
}

// Create cost derivative functions - PDF Reference: Section 8.7.2
void iLQR::createCostDerivativeFunctions(const casadi::SX& stage_cost, const std::vector<casadi::SX>& all_vars) {
    // First derivatives
    for (int k = 0; k <= N_; k++) {
        cost_grad_x_[k] = casadi::Function("cost_grad_x_" + std::to_string(k), all_vars, {casadi::SX::gradient(stage_cost, x_sym_[k])});
        
        if (k < N_) {
            cost_grad_u_[k] = casadi::Function("cost_grad_u_" + std::to_string(k), all_vars, {casadi::SX::gradient(stage_cost, u_sym_[k])});
            cost_grad_y_[k] = casadi::Function("cost_grad_y_" + std::to_string(k), all_vars, {casadi::SX::gradient(stage_cost, x_sym_[k+1])});
            
            if (k < N_ - 1) {
                cost_grad_v_[k] = casadi::Function("cost_grad_v_" + std::to_string(k), all_vars, {casadi::SX::gradient(stage_cost, u_sym_[k+1])});
            }
        }
    }
    
    // Second derivatives
    for (int k = 0; k <= N_; k++) {
        cost_hess_xx_[k] = casadi::Function("cost_hess_xx_" + std::to_string(k), all_vars, {casadi::SX::hessian(stage_cost, x_sym_[k])});
        
        if (k < N_) {
            cost_hess_uu_[k] = casadi::Function("cost_hess_uu_" + std::to_string(k), all_vars, {casadi::SX::hessian(stage_cost, u_sym_[k])});
            cost_hess_yy_[k] = casadi::Function("cost_hess_yy_" + std::to_string(k), all_vars, {casadi::SX::hessian(stage_cost, x_sym_[k+1])});
            
            if (k < N_ - 1) {
                cost_hess_vv_[k] = casadi::Function("cost_hess_vv_" + std::to_string(k), all_vars, {casadi::SX::hessian(stage_cost, u_sym_[k+1])});
            }
        }
    }
    
    // Cross derivatives
    for (int k = 0; k < N_; k++) {
        cost_hess_xu_[k] = casadi::Function("cost_hess_xu_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k]), u_sym_[k])});
        cost_hess_ux_[k] = casadi::Function("cost_hess_ux_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k]), x_sym_[k])});
        cost_hess_xy_[k] = casadi::Function("cost_hess_xy_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k]), x_sym_[k+1])});
        cost_hess_yx_[k] = casadi::Function("cost_hess_yx_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k+1]), x_sym_[k])});
        cost_hess_uy_[k] = casadi::Function("cost_hess_uy_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k]), x_sym_[k+1])});
        cost_hess_yu_[k] = casadi::Function("cost_hess_yu_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k+1]), u_sym_[k])});
        
        if (k < N_ - 1) {
            cost_hess_xv_[k] = casadi::Function("cost_hess_xv_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k]), u_sym_[k+1])});
            cost_hess_vx_[k] = casadi::Function("cost_hess_vx_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k+1]), x_sym_[k])});
            cost_hess_uv_[k] = casadi::Function("cost_hess_uv_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k]), u_sym_[k+1])});
            cost_hess_vu_[k] = casadi::Function("cost_hess_vu_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k+1]), u_sym_[k])});
            cost_hess_yv_[k] = casadi::Function("cost_hess_yv_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, x_sym_[k+1]), u_sym_[k+1])});
            cost_hess_vy_[k] = casadi::Function("cost_hess_vy_" + std::to_string(k), all_vars, {casadi::SX::jacobian(casadi::SX::gradient(stage_cost, u_sym_[k+1]), x_sym_[k+1])});
        }
    }
    std::cout << "computed the cost derivatives" << std::endl;
}

// Add to createSymbolicFramework() after cost derivatives
void iLQR::createConstraintDerivativeFunctions(const casadi::SX& constraint_cost, 
                                               const std::vector<casadi::SX>& all_vars) {
    for (int k = 0; k < N_; k++) {
        // Constraint gradient w.r.t. controls (most important)
        constraint_grad_u_[k] = casadi::Function("constraint_grad_u_" + std::to_string(k), 
            all_vars, {casadi::SX::gradient(constraint_cost, u_sym_[k])});
        constraint_grad_x_[k] = casadi::Function("constraint_grad_x_" + std::to_string(k), 
            all_vars, {casadi::SX::gradient(constraint_cost, x_sym_[k])});
        constraint_hess_uu_[k] = casadi::Function("constraint_hess_uu_" + std::to_string(k), 
            all_vars, {casadi::SX::hessian(constraint_cost, u_sym_[k])});
        constraint_hess_xx_[k] = casadi::Function("constraint_hess_xx_" + std::to_string(k), 
            all_vars, {casadi::SX::hessian(constraint_cost, x_sym_[k])});
        constraint_hess_ux_[k] = casadi::Function("constraint_hess_ux_" + std::to_string(k), 
            all_vars, {casadi::SX::jacobian(casadi::SX::gradient(constraint_cost, u_sym_[k]), x_sym_[k])});
    }
    std::cout << "computed the constraint derivatives" << std::endl;
}

// PDF Reference: Section 8.7.2 "Second order expansion of L(x,u,y,v)"
void iLQR::quadratizeCost() {
    std::vector<casadi::DM> all_inputs;
    collectAllInputs(all_inputs);
    
    // Evaluate all derivatives numerically
    for (int k = 0; k <= N_; k++) {
        // State derivatives ([0,N])
        L_x_[k] = cost_grad_x_[k](all_inputs)[0];
        L_xx_[k] = cost_hess_xx_[k](all_inputs)[0];
        
        if (k < N_) {            // [0,N)
            // Current control derivatives
            L_u_[k] = cost_grad_u_[k](all_inputs)[0];
            L_uu_[k] = cost_hess_uu_[k](all_inputs)[0];
            
            // Next state derivatives (y = x_{k+1})
            L_y_[k] = cost_grad_y_[k](all_inputs)[0];
            L_yy_[k] = cost_hess_yy_[k](all_inputs)[0];
            
            // Cross derivatives: current state-control
            L_xu_[k] = cost_hess_xu_[k](all_inputs)[0];
            L_ux_[k] = cost_hess_ux_[k](all_inputs)[0];
            
            // Cross derivatives: current state - next state
            L_xy_[k] = cost_hess_xy_[k](all_inputs)[0];
            L_yx_[k] = cost_hess_yx_[k](all_inputs)[0];
            
            // Cross derivatives: current control - next state
            L_uy_[k] = cost_hess_uy_[k](all_inputs)[0];
            L_yu_[k] = cost_hess_yu_[k](all_inputs)[0];
            
            // Next control derivatives (v = u_{k+1})
            if (k < N_ - 1) {
                // [0, N-1)
                L_v_[k] = cost_grad_v_[k](all_inputs)[0];
                L_vv_[k] = cost_hess_vv_[k](all_inputs)[0];
                
                // Cross derivatives: current state - next control
                L_xv_[k] = cost_hess_xv_[k](all_inputs)[0];
                L_vx_[k] = cost_hess_vx_[k](all_inputs)[0];
                
                // Cross derivatives: current control - next control
                L_uv_[k] = cost_hess_uv_[k](all_inputs)[0];
                L_vu_[k] = cost_hess_vu_[k](all_inputs)[0];
                
                // Cross derivatives: next state - next control
                L_yv_[k] = cost_hess_yv_[k](all_inputs)[0];
                L_vy_[k] = cost_hess_vy_[k](all_inputs)[0];
            }
        }
    }
    linearizeConstraints(all_inputs);
}

// Collect all numerical inputs for CasADi function evaluation
void iLQR::collectAllInputs(std::vector<casadi::DM>& all_inputs) {
    all_inputs.clear();
    all_inputs.insert(all_inputs.end(), x_traj_.begin(), x_traj_.end());
    all_inputs.insert(all_inputs.end(), x_ref_.begin(), x_ref_.end());
    all_inputs.insert(all_inputs.end(), u_traj_.begin(), u_traj_.end());
    all_inputs.insert(all_inputs.end(), u_ref_.begin(), u_ref_.end());
    all_inputs.insert(all_inputs.end(), u_art_.begin(), u_art_.end());
    all_inputs.insert(all_inputs.end(), lambda_.begin(), lambda_.end());
    all_inputs.insert(all_inputs.end(), mu_.begin(), mu_.end());
}

// Add to quadratizeCost() function
void iLQR::linearizeConstraints(const std::vector<casadi::DM>& all_inputs) {    
    for (int k = 0; k < N_; k++) {
        C_u_[k] = constraint_grad_u_[k](all_inputs)[0];
        C_x_[k] = constraint_grad_x_[k](all_inputs)[0];
        C_uu_[k] = constraint_hess_uu_[k](all_inputs)[0];
        C_xx_[k] = constraint_hess_xx_[k](all_inputs)[0];
        C_ux_[k] = constraint_hess_ux_[k](all_inputs)[0];
    }
}

// PDF Reference: Section 8.3, Equation (43)
void iLQR::linearizeDynamics() {
    for (int k = 0; k < N_; k++) {
        std::vector<casadi::DM> input;
        
        if (k < N_ - 1) {
            input = {x_traj_[k], u_traj_[k], u_traj_[k + 1]};
        } else {
            input = {x_traj_[k], u_traj_[k], u_traj_[k]};  // Boundary condition
        }
        
        // Direct assignment - much cleaner
        A_[k] = dynamics_jac_A_(input)[0];
        B_[k] = dynamics_jac_B_(input)[0];
        C_[k] = dynamics_jac_C_(input)[0];
    }
}

// Complete backward pass with constraint handling
void iLQR::backwardPass() {
    // Initialize terminal cost-to-go
    casadi::DM x_error_final = x_traj_[N_] - x_ref_[N_];
    
    if (S_[N_].rows() != nx_ + nu_ || S_[N_].columns() != nx_ + nu_) S_[N_] = casadi::DM::zeros(nx_ + nu_, nx_ + nu_);
    if (s_[N_].rows() != nx_ + nu_) s_[N_] = casadi::DM::zeros(nx_ + nu_);
    
    // Fill state portion
    S_[N_](casadi::Slice(0, nx_), casadi::Slice(0, nx_)) = Qf_;
    s_[N_](casadi::Slice(0, nx_)) = casadi::DM::mtimes(Qf_, x_error_final);
    
    // Backward loop from N-1 to 0
    for (int k = N_ - 1; k >= 0; k--) {
        computeQderivatives(k);
        computeGains(k);
        updateValueFunction(k);
    }
}

// Compute Q-derivatives with G+H formulation and constraints
void iLQR::computeQderivatives(int k) {
    if (k >= N_ || k < 0) return;
    
    // Extract cost-to-go from next time step
    casadi::DM S_next = S_[k + 1];
    casadi::DM s_next = s_[k + 1];
    
    // Extract blocks from (n+m) × (n+m) cost-to-go matrix
    casadi::DM Syy = S_next(casadi::Slice(0, nx_), casadi::Slice(0, nx_));
    casadi::DM Syv = S_next(casadi::Slice(0, nx_), casadi::Slice(nx_, nx_ + nu_));
    casadi::DM Svy = S_next(casadi::Slice(nx_, nx_ + nu_), casadi::Slice(0, nx_));
    casadi::DM Svv = S_next(casadi::Slice(nx_, nx_ + nu_), casadi::Slice(nx_, nx_ + nu_));
    
    casadi::DM sy = s_next(casadi::Slice(0, nx_));
    casadi::DM sv = s_next(casadi::Slice(nx_, nx_ + nu_));
    
    // Compute G-terms (from cost-to-go propagation)
    casadi::DM Gxx = casadi::DM::mtimes({A_[k].T(), Syy, A_[k]});
    casadi::DM Guu = casadi::DM::mtimes({B_[k].T(), Syy, B_[k]});
    casadi::DM Gxu = casadi::DM::mtimes({A_[k].T(), Syy, B_[k]});
    casadi::DM Gx = casadi::DM::mtimes(A_[k].T(), sy);
    casadi::DM Gu = casadi::DM::mtimes(B_[k].T(), sy);
    
    // Compute H-terms (from stage cost)
    casadi::DM Hxx = L_xx_[k] + casadi::DM::mtimes(L_xy_[k], A_[k]) + 
                     casadi::DM::mtimes(A_[k].T(), L_yx_[k]) + 
                     casadi::DM::mtimes({A_[k].T(), L_yy_[k], A_[k]});
    
    casadi::DM Huu = L_uu_[k] + casadi::DM::mtimes(L_uy_[k], B_[k]) + 
                     casadi::DM::mtimes(B_[k].T(), L_yu_[k]) + 
                     casadi::DM::mtimes({B_[k].T(), L_yy_[k], B_[k]});
    
    casadi::DM Hxu = L_xu_[k] + casadi::DM::mtimes(L_xy_[k], B_[k]) + 
                     casadi::DM::mtimes(A_[k].T(), L_yu_[k]) + 
                     casadi::DM::mtimes({A_[k].T(), L_yy_[k], B_[k]});
    
    casadi::DM Hx = L_x_[k] + casadi::DM::mtimes(A_[k].T(), L_y_[k]);
    casadi::DM Hu = L_u_[k] + casadi::DM::mtimes(B_[k].T(), L_y_[k]);
    
    // Handle v-related terms for k < N-1
    casadi::DM Gvv, Gxv, Guv, Gv, Hvv, Hxv, Huv, Hv;
    if (k < N_ - 1) {
        Gvv = casadi::DM::mtimes({C_[k].T(), Syy, C_[k]}) + casadi::DM::mtimes(C_[k].T(), Syv) + casadi::DM::mtimes(Svy, C_[k]) + Svv;
        Gxv = casadi::DM::mtimes({A_[k].T(), Syy, C_[k]}) + casadi::DM::mtimes(A_[k].T(), Syv);
        Guv = casadi::DM::mtimes({B_[k].T(), Syy, C_[k]}) + casadi::DM::mtimes(B_[k].T(), Syv);
        Gv = casadi::DM::mtimes(C_[k].T(), sy) + sv;
        
        Hvv = L_vv_[k] + casadi::DM::mtimes(L_vy_[k], C_[k]) + 
              casadi::DM::mtimes(C_[k].T(), L_yv_[k]) + 
              casadi::DM::mtimes({C_[k].T(), L_yy_[k], C_[k]});
        Hxv = L_xv_[k] + casadi::DM::mtimes(L_xy_[k], C_[k]) + 
              casadi::DM::mtimes(A_[k].T(), L_yv_[k]) + 
              casadi::DM::mtimes({A_[k].T(), L_yy_[k], C_[k]});
        Huv = L_uv_[k] + casadi::DM::mtimes(L_uy_[k], C_[k]) + 
              casadi::DM::mtimes(B_[k].T(), L_yv_[k]) + 
              casadi::DM::mtimes({B_[k].T(), L_yy_[k], C_[k]});
        Hv = L_v_[k] + casadi::DM::mtimes(C_[k].T(), L_y_[k]);
    } else {
        // Edge case: k = N-1, no v-derivatives
        Gvv = casadi::DM::zeros(nu_, nu_);
        Gxv = casadi::DM::zeros(nx_, nu_);
        Guv = casadi::DM::zeros(nu_, nu_);
        Gv = casadi::DM::zeros(nu_);
        Hvv = casadi::DM::zeros(nu_, nu_);
        Hxv = casadi::DM::zeros(nx_, nu_);
        Huv = casadi::DM::zeros(nu_, nu_);
        Hv = casadi::DM::zeros(nu_);
    }
    
    // Complete Q-function derivatives (G + H + constraints)
    Q_x_[k] = Gx + Hx + C_x_[k];
    Q_u_[k] = Gu + Hu + C_u_[k];
    Q_xx_[k] = Gxx + Hxx + C_xx_[k];
    Q_uu_[k] = Guu + Huu + C_uu_[k];
    Q_ux_[k] = Gxu.T() + Hxu.T() + C_ux_[k];
    
    if (k < N_ - 1) {
        Q_v_[k] = Gv + Hv;
        Q_vv_[k] = Gvv + Hvv;
        Q_vx_[k] = Gxv.T() + Hxv.T();
        Q_uv_[k] = Guv + Huv;
    }
}

// Compute FOH gains with edge case handling
void iLQR::computeGains(int k) {
    if (k >= N_ || k < 0) return;
    
    applyRegularization(k);
    
    if (k == N_ - 1) {
        // Edge case: Final time step, no v-terms
        K_[k] = -casadi::DM::solve(Q_uu_[k], Q_ux_[k]);
        d_[k] = -casadi::DM::solve(Q_uu_[k], Q_u_[k]);
        b_[k] = casadi::DM::zeros(nu_, nu_);
    } else {
        // Standard FOH gains
        K_[k] = -casadi::DM::solve(Q_vv_[k], Q_vx_[k]);
        b_[k] = -casadi::DM::solve(Q_vv_[k], Q_uv_[k].T());
        d_[k] = -casadi::DM::solve(Q_vv_[k], Q_v_[k]);
    }
}

// Update value function with Q-star formulation
void iLQR::updateValueFunction(int k) {
    if (k >= N_ || k < 0) return;
    
    if (k == N_ - 1) {
        // Final time step uses standard LQR Q-star formulation
        casadi::DM Qxx_star = Q_xx_[k] + casadi::DM::mtimes(Q_ux_[k].T(), K_[k]) + 
                             casadi::DM::mtimes(K_[k].T(), Q_ux_[k]) + 
                             casadi::DM::mtimes({K_[k].T(), Q_uu_[k], K_[k]});
        
        casadi::DM Quu_star = Q_uu_[k];
        casadi::DM Qux_star = Q_ux_[k] + casadi::DM::mtimes(Q_uu_[k], K_[k]);
        
        casadi::DM Qx_star = Q_x_[k] + casadi::DM::mtimes(Q_ux_[k].T(), d_[k]) + 
                            casadi::DM::mtimes(K_[k].T(), Q_u_[k]) + 
                            casadi::DM::mtimes({K_[k].T(), Q_uu_[k], d_[k]});
        
        casadi::DM Qu_star = Q_u_[k] + casadi::DM::mtimes(Q_uu_[k], d_[k]);
        
        // Assemble (n+m) × (n+m) cost-to-go matrix
        S_[k](casadi::Slice(0, nx_), casadi::Slice(0, nx_)) = Qxx_star;
        S_[k](casadi::Slice(0, nx_), casadi::Slice(nx_, nx_ + nu_)) = Qux_star.T();
        S_[k](casadi::Slice(nx_, nx_ + nu_), casadi::Slice(0, nx_)) = Qux_star;
        S_[k](casadi::Slice(nx_, nx_ + nu_), casadi::Slice(nx_, nx_ + nu_)) = Quu_star;
        
        s_[k](casadi::Slice(0, nx_)) = Qx_star;
        s_[k](casadi::Slice(nx_, nx_ + nu_)) = Qu_star;
    } else {
        // Standard FOH Q-star computation
        casadi::DM Qxx_star = Q_xx_[k] + casadi::DM::mtimes(Q_vx_[k].T(), K_[k]) + 
                             casadi::DM::mtimes(K_[k].T(), Q_vx_[k]) + 
                             casadi::DM::mtimes({K_[k].T(), Q_vv_[k], K_[k]});
        
        casadi::DM Quu_star = Q_uu_[k] + casadi::DM::mtimes(Q_uv_[k], b_[k]) + 
                             casadi::DM::mtimes(b_[k].T(), Q_uv_[k].T()) + 
                             casadi::DM::mtimes({b_[k].T(), Q_vv_[k], b_[k]});
        
        casadi::DM Qux_star = Q_ux_[k] + casadi::DM::mtimes(Q_uv_[k], K_[k]) + 
                             casadi::DM::mtimes(b_[k].T(), Q_vx_[k]) + 
                             casadi::DM::mtimes({b_[k].T(), Q_vv_[k], K_[k]});
        
        casadi::DM Qx_star = Q_x_[k] + casadi::DM::mtimes(Q_vx_[k].T(), d_[k]) + 
                            casadi::DM::mtimes(K_[k].T(), Q_v_[k]) + 
                            casadi::DM::mtimes({K_[k].T(), Q_vv_[k], d_[k]});
        
        casadi::DM Qu_star = Q_u_[k] + casadi::DM::mtimes(Q_uv_[k], d_[k]) + 
                            casadi::DM::mtimes(b_[k].T(), Q_v_[k]) + 
                            casadi::DM::mtimes({b_[k].T(), Q_vv_[k], d_[k]});
        
        // Assemble (n+m) × (n+m) cost-to-go matrix
        S_[k](casadi::Slice(0, nx_), casadi::Slice(0, nx_)) = Qxx_star;
        S_[k](casadi::Slice(0, nx_), casadi::Slice(nx_, nx_ + nu_)) = Qux_star.T();
        S_[k](casadi::Slice(nx_, nx_ + nu_), casadi::Slice(0, nx_)) = Qux_star;
        S_[k](casadi::Slice(nx_, nx_ + nu_), casadi::Slice(nx_, nx_ + nu_)) = Quu_star;
        
        s_[k](casadi::Slice(0, nx_)) = Qx_star;
        s_[k](casadi::Slice(nx_, nx_ + nu_)) = Qu_star;
    }
}

void iLQR::applyRegularization(int k) {
    casadi::DM I = casadi::DM::eye(nu_);
    double adaptive_reg = std::max(reg_, 1e-4);  // Minimum regularization
    
    if (k == N_ - 1) {
        Q_uu_[k] += adaptive_reg * I;
        
        // Check condition number and adapt
        double det = static_cast<double>(casadi::DM::det(Q_uu_[k]));
        if (std::abs(det) < 1e-10) {
            adaptive_reg *= 100;                // Aggressive increase
            Q_uu_[k] += adaptive_reg * I;
        }
    } else {
        Q_uu_[k] += adaptive_reg * I;
        Q_vv_[k] += adaptive_reg * I;
    }
}

// Forward pass with line search
bool iLQR::forwardPass() {
    double best_cost = current_cost_;
    double best_alpha = 0.0;
    bool found_improvement = false;
    
    // Store original trajectory
    auto x_orig = x_traj_;
    auto u_orig = u_traj_;
    auto u_art_orig = u_art_;
    
    for (double alpha : alpha_schedule_) {
        // Reset to original trajectory for each test
        x_traj_ = x_orig;
        u_traj_ = u_orig;
        u_art_ = u_art_orig;
        
        double cost = simulateTrajectory(alpha);
        // std::cout << "cost: " << cost << " alpha: " << alpha << std::endl;
        if (cost < best_cost) {
            best_cost = cost;
            best_alpha = alpha;
            found_improvement = true;
        }
    }
    
    if (found_improvement) {
        // Apply best trajectory
        simulateTrajectory(best_alpha);
        current_cost_ = best_cost;
    }
    
    return found_improvement;
}

// Simulate trajectory with FOH gains
double iLQR::simulateTrajectory(double alpha) {
    std::vector<casadi::DM> x_new(N_ + 1);
    std::vector<casadi::DM> u_new(N_);
    std::vector<casadi::DM> u_art_new(N_);
    
    x_new[0] = x_traj_[0];
    
    // First control update (no previous control for FOH)
    u_new[0] = u_traj_[0] + alpha * d_[0];
    
    // Forward simulation with FOH gains
    for (int k = 0; k < N_; k++) {
        casadi::DM dx = x_new[k] - x_traj_[k];
        
        if (k > 0) {
            // FOH control update with feedthrough gain
            casadi::DM du_prev = u_new[k-1] - u_traj_[k-1];
            u_new[k] = u_traj_[k] + casadi::DM::mtimes(K_[k], dx) + casadi::DM::mtimes(b_[k], du_prev) + alpha * d_[k];
        }
        
        // Compute artificial control (scaled by alpha)
        casadi::DM x_natural;
        if (k < N_ - 1) {
            x_natural = foh_dynamics_func_(std::vector<casadi::DM>{x_new[k], u_new[k], u_traj_[k+1]})[0];
        } else {
            x_natural = foh_dynamics_func_(std::vector<casadi::DM>{x_new[k], u_new[k], u_new[k]})[0];
        }
        
        u_art_new[k] = alpha * (x_traj_[k+1] - x_natural);
        
        // Apply FOH dynamics with artificial control
        if (k < N_) {
            x_new[k+1] = x_natural + u_art_new[k];
        }
    }
    
    // Compute cost of new trajectory
    double new_cost = evaluateTrajectory(x_new, u_new, u_art_new);
    
    // Update trajectories if cost improved
    if (new_cost < current_cost_) {
        x_traj_ = x_new;
        u_traj_ = u_new;
        u_art_ = u_art_new;
    }
    
    return new_cost;
}

// Evaluate trajectory cost
double iLQR::evaluateTrajectory(const std::vector<casadi::DM>& x_traj,
                               const std::vector<casadi::DM>& u_traj,
                               const std::vector<casadi::DM>& u_art) {
    // Prepare input vector for cost function
    std::vector<casadi::DM> all_inputs;
    all_inputs.insert(all_inputs.end(), x_traj.begin(), x_traj.end());
    all_inputs.insert(all_inputs.end(), x_ref_.begin(), x_ref_.end());
    all_inputs.insert(all_inputs.end(), u_traj.begin(), u_traj.end());
    all_inputs.insert(all_inputs.end(), u_ref_.begin(), u_ref_.end());
    all_inputs.insert(all_inputs.end(), u_art.begin(), u_art.end());
    all_inputs.insert(all_inputs.end(), lambda_.begin(), lambda_.end());
    all_inputs.insert(all_inputs.end(), mu_.begin(), mu_.end());
    
    return static_cast<double>(total_cost_func_(all_inputs)[0]);
}

// Main solve function
bool iLQR::solve(const std::vector<casadi::DM>& x_guess,
                const std::vector<casadi::DM>& x_ref,
                const std::vector<casadi::DM>& u_ref) {
    
    initializeTrajectory(x_guess);
    if (!x_ref.empty()) x_ref_ = x_ref;
    if (!u_ref.empty()) u_ref_ = u_ref;
    
    current_cost_ = computeCost();
    // previous_cost_ = current_cost_;
    
    for (int iter = 0; iter < max_iter_; iter++) {
        previous_cost_ = current_cost_;

        linearizeDynamics();
        quadratizeCost();
        backwardPass();
        if (current_cost_ > 1e6) {  // Numerical explosion
            return false;
        }

        std::cout << "cost vefore fwd pass: " << current_cost_ << std::endl;
        if (!forwardPass()) {
            // break;
        }
        
        if (checkConvergence()) {
            converged_ = true;
            casadi::DM contact_forces = getContactForces();
            std::cout << "Converged after " << iter + 1 << " iterations!" << std::endl;
            break;
        }
    }
    std::cout << "Previous cost: " << previous_cost_ << ", Current cost: " << current_cost_ << std::endl;
    
    return true;
}

// Initialize trajectory from initial guess
void iLQR::initializeTrajectory(const std::vector<casadi::DM>& x_guess) {
    if (x_guess.size() != x_traj_.size()) {
        std::cerr << "Error: Provided initial trajectory guess size (" << x_guess.size()
                  << ") does not match required size (" << x_traj_.size() << ")" << std::endl;
        throw std::runtime_error("Initial trajectory guess size mismatch");
    }
    
    for (int k = 0; k < static_cast<double>(x_guess.size()); ++k) {
        x_traj_[k] = x_guess[k];
    }
    
    // Compute feasible controls using robot dynamics
    for (int k = 0; k < N_; k++) {
        casadi::DM x_current = x_traj_[k];
        casadi::DM x_next = x_traj_[k + 1];
        
        // Extract q and qd from current and next states
        casadi::DM q_current = x_current(casadi::Slice(0, robot_utils_->nq()));
        casadi::DM qd_current = x_current(casadi::Slice(robot_utils_->nq(), robot_utils_->nx()));
        casadi::DM q_next = x_next(casadi::Slice(0, robot_utils_->nq()));
        casadi::DM qd_next = x_next(casadi::Slice(robot_utils_->nq(), robot_utils_->nx()));
        
        // Compute required acceleration using finite differences
        casadi::DM qdd_required = (qd_next - qd_current) / dt_;
        
        // Use inverse dynamics to compute required torques
        casadi::DM u_required = computeInverseDynamics(q_current, qd_current, qdd_required);
        
        u_traj_[k] = u_required;
        u_art_[k] = casadi::DM::zeros(nx_);              // No artificial control needed for consistent trajectory
    }
}


// Dynamics helper
casadi::DM iLQR::humanoidDynamics(const casadi::DM& x, const casadi::DM& u) {

    return foh_dynamics_func_(std::vector<casadi::DM>{x, u, u})[0];
}


// Compute current trajectory cost
double iLQR::computeCost() {
    std::vector<casadi::DM> all_inputs;
    collectAllInputs(all_inputs);
    return static_cast<double>(total_cost_func_(all_inputs)[0]);
}

casadi::DM iLQR::computeInverseDynamics(const casadi::DM& q, const casadi::DM& qd, const casadi::DM& qdd) {
    if (!inv_dyn_func_cached_) {
        casadi::SX q_sym = casadi::SX::sym("q_temp", robot_utils_->nq());
        casadi::SX qd_sym = casadi::SX::sym("qd_temp", robot_utils_->nv());
        casadi::SX qdd_sym = casadi::SX::sym("qdd_temp", robot_utils_->nv());
        
        casadi::SX tau_expr = robot_utils_->symInverseDynamics(q_sym, qd_sym, qdd_sym);
        cached_inv_dyn_func_ = casadi::Function("cached_inverse_dynamics", 
                                               {q_sym, qd_sym, qdd_sym}, {tau_expr});
        inv_dyn_func_cached_ = true;
    }
    
    return cached_inv_dyn_func_(std::vector<casadi::DM>{q, qd, qdd})[0];
}

casadi::DM iLQR::getContactForces() const {
    return robot_utils_->computeContactForcesNumeric(x_traj_, u_traj_);
}


// Update constraint multipliers
void iLQR::updateMultipliers() {
    for (int k = 0; k < N_; k++) {
        casadi::DM torque = u_traj_[k](0);
        
        // Upper bound constraint: u - u_max <= 0
        double c_upper = static_cast<double>(torque) - u_max_;
        if (c_upper > 0 || static_cast<double>(lambda_[k](0)) > 0) {
            lambda_[k](0) = std::max(0.0, static_cast<double>(lambda_[k](0)) + 
                                     static_cast<double>(mu_[k](0)) * c_upper);
        }
        
        // Lower bound constraint: u_min - u <= 0
        double c_lower = u_min_ - static_cast<double>(torque);
        if (c_lower > 0 || static_cast<double>(lambda_[k](1)) > 0) {
            lambda_[k](1) = std::max(0.0, static_cast<double>(lambda_[k](1)) + 
                                     static_cast<double>(mu_[k](1)) * c_lower);
        }
    }
}

// Check convergence criteria
bool iLQR::checkConvergence() {    
    double cost_reduction = std::abs(previous_cost_ - current_cost_) / 
                           std::max(std::abs(previous_cost_), 1.0);
    
    // Also check gradient norm for better convergence criterion
    return cost_reduction < tolerance_;
}
