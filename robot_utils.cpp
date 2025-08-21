#include "robot_utils.hpp"
#include <iostream>
#include <stdexcept>

using namespace pinocchio;

RobotUtils::RobotUtils(const std::string& urdf_path) {
    // Load robot model with floating base (corrected approach)
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);
    
    // Set dimensions
    nq_ = model_.nq;
    nv_ = model_.nv;
    nx_ = nq_ + nv_;
    nu_ = nv_;
    
    // Initialize joint limits
    lower_limits_.resize(nq_);
    upper_limits_.resize(nq_);
    valid_limits_.resize(nq_, false);
    
    for (int i = 0; i < model_.nq; ++i) {
        double lower = model_.lowerPositionLimit[i];
        double upper = model_.upperPositionLimit[i];
        
        // Check if limits are finite and valid
        bool has_valid_limits = (lower > -1e10 && upper < 1e10 && lower < upper);
        
        if (has_valid_limits) {
            lower_limits_[i] = lower;
            upper_limits_[i] = upper;
            valid_limits_[i] = true;
        } else {
            // Set safe default limits for invalid joints
            lower_limits_[i] = -3.14;  // -180 degrees
            upper_limits_[i] = 3.14;   // +180 degrees  
            valid_limits_[i] = false;
        }
    }
    
    // Print summary of valid limits
    int valid_count = 0;
    for (int i = 7; i < nq_; ++i) {  // Skip floating base
        if (valid_limits_[i]) valid_count++;
    }
    std::cout << "Found " << valid_count << " joints with valid limits out of " 
              << (nq_ - 7) << " actuated joints" << std::endl;
    // Initialize default end effectors
    ee_names_ = {"left_ankle_link", "right_ankle_link"};
    
    std::cout << "RobotUtils initialized: nq =" << nq_ << ", nv=" << nv_ 
              << ", nx =" << nx_ << ", nu =" << nu_ << std::endl;
}

::casadi::SX RobotUtils::symFloatingBaseDynamics(const ::casadi::SX& x, const ::casadi::SX& u) const {
    // Extract q and qd from state vector
    ::casadi::SX q = x(::casadi::Slice(0, nq_));
    ::casadi::SX qd = x(::casadi::Slice(nq_, nx_));
    
    typedef ::casadi::SX Scalar;
    typedef ModelTpl<Scalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVector;
    typedef typename ADModel::TangentVectorType TangentVector;
    
    // Cast model to autodiff version
    ADModel ad_model = model_.cast<Scalar>();
    ADData ad_data(ad_model);
    
    // Convert ::casadi variables to Pinocchio types
    ConfigVector q_ad(nq_);
    TangentVector qd_ad(nv_);
    TangentVector tau_ad(nv_);
    
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q(i);
    }
    for (int i = 0; i < nv_; ++i) {
        qd_ad(i) = qd(i);
        tau_ad(i) = u(i);           // Includes zeros for floating base
    }
    
    pinocchio::crba(ad_model, ad_data, q_ad);
    ::casadi::SX M = ::casadi::SX::zeros(nv_, nv_);
    for (int i = 0; i < nv_; ++i) {
        for (int j = 0; j < nv_; ++j) {
            M(i, j) = ad_data.M(i, j);
        }
    }
    
    // Compute bias forces h(q,qd) = C(q,qd)*qd + g(q)
    pinocchio::rnea(ad_model, ad_data, q_ad, qd_ad, TangentVector::Zero(nv_));
    ::casadi::SX h = ::casadi::SX::zeros(nv_, 1);
    for (int i = 0; i < nv_; ++i) {
        h(i) = ad_data.tau(i);
    }
    
    // Compute soft contact forces
    ::casadi::SX F_contact = computeSoftContactForces(q, qd);
    ::casadi::SX J_contact = computeContactJacobians(q);
    
    // Control vector (includes zeros for floating base)
    ::casadi::SX tau_vec = ::casadi::SX::zeros(nv_, 1);
    for (int i = 0; i < nv_; ++i) {
        tau_vec(i) = tau_ad(i);
    }
    
    // M(q)qdd + h(q,qd) = tau + J_contact^T * F_contact
    ::casadi::SX contact_forces_mapped = ::casadi::SX::mtimes(J_contact.T(), F_contact);
    ::casadi::SX qdd = ::casadi::SX::solve(M, tau_vec - h + contact_forces_mapped);
    ::casadi::SX qdot = computeConfigurationDerivative(q, qd);
    
    // Return state derivative [qd, qdd]
    return ::casadi::SX::vertcat({qdot, qdd});
}

::casadi::SX RobotUtils::symCostStage(const ::casadi::SX& x, const ::casadi::SX& u,
                                     const ::casadi::SX& x_ref, const ::casadi::SX& u_ref) const {
    ::casadi::SX cost = 0;
    
    // Extract components
    ::casadi::SX q = x(::casadi::Slice(0, nq_));
    ::casadi::SX qd = x(::casadi::Slice(nq_, nx_));
    ::casadi::SX q_ref = x_ref(::casadi::Slice(0, nq_));
    ::casadi::SX qd_ref = x_ref(::casadi::Slice(nq_, nx_));

    // Base position tracking (first 3 elements)
    ::casadi::SX base_pos_error = q(::casadi::Slice(0, 3)) - q_ref(::casadi::Slice(0, 3));
    cost += weights.Q_base_pos * ::casadi::SX::dot(base_pos_error, base_pos_error);

    // Base orientation tracking (quaternion: elements 3-6)
    ::casadi::SX base_ori_error = q(::casadi::Slice(3, 7)) - q_ref(::casadi::Slice(3, 7));
    cost += weights.Q_base_ori * ::casadi::SX::dot(base_ori_error, base_ori_error);

    // Joint tracking (remaining elements)
    if (nq_ > 7) {
        ::casadi::SX joint_error = q(::casadi::Slice(7, nq_)) - q_ref(::casadi::Slice(7, nq_));
        cost += weights.Q_joints * ::casadi::SX::dot(joint_error, joint_error);
    }

    // Velocity tracking
    ::casadi::SX vel_error = qd - qd_ref;
    cost += weights.Q_velocity * ::casadi::SX::dot(vel_error, vel_error);

    // Control effort (only actuated joints) - ADD DIMENSION CHECK
    if (u.size1() > 0 && u_ref.size1() > 0) {  // Check if control vectors are non-empty
        if (nv_ > 6 && u.size1() >= nv_ && u_ref.size1() >= nv_) {
            ::casadi::SX u_actuated = u(::casadi::Slice(6, nv_));
            ::casadi::SX u_ref_actuated = u_ref(::casadi::Slice(6, nv_));
            ::casadi::SX control_error = u_actuated - u_ref_actuated;
            cost += weights.R_torque * ::casadi::SX::dot(control_error, control_error);
        } else if (u.size1() == u_ref.size1()) {
            // Handle case where full control vector is provided
            ::casadi::SX control_error = u - u_ref;
            cost += weights.R_torque * ::casadi::SX::dot(control_error, control_error);
        }
    }

    // Angular momentum regulation
    ::casadi::SX L = angularMomentum(q, qd);
    cost += weights.Q_ang_momentum * ::casadi::SX::dot(L, L);

    return cost;
}

::casadi::SX RobotUtils::symConstraintPenalty(const ::casadi::SX& x, const ::casadi::SX& u) const {
    ::casadi::SX penalty = 0;
    
    ::casadi::SX q = x(::casadi::Slice(0, nq_));
    ::casadi::SX qd = x(::casadi::Slice(nq_, nx_));
    
    // Joint position limits (only for joints with valid limits)
    for (int i = 7; i < nq_; ++i) {             // Skip floating base
        if (valid_limits_[i]) {                 // Only apply constraints to joints with valid limits
            ::casadi::SX pos_upper_viol = ::casadi::SX::fmax(0, q(i) - upper_limits_[i]);
            ::casadi::SX pos_lower_viol = ::casadi::SX::fmax(0, lower_limits_[i] - q(i));
            penalty += weights.penalty_joint * (pos_upper_viol * pos_upper_viol + 
                                               pos_lower_viol * pos_lower_viol);
        }
    }
    
    // Velocity limits (apply to all actuated joints with reasonable defaults)
    for (int i = 6; i < std::min(nv_, nq_); ++i) { 
        double vel_limit = 10.0; 
        if (i < nv_ && model_.velocityLimit[i] < 1e10 && model_.velocityLimit[i] > 0) {
            vel_limit = model_.velocityLimit[i];
        }
        
        ::casadi::SX vel_upper_viol = ::casadi::SX::fmax(0, qd(i) - vel_limit);
        ::casadi::SX vel_lower_viol = ::casadi::SX::fmax(0, -vel_limit - qd(i));
        penalty += weights.penalty_joint * (vel_upper_viol * vel_upper_viol + 
                                           vel_lower_viol * vel_lower_viol);
    }
    
    // Torque limits (apply to all actuated joints)
    for (int i = 6; i < nu_; ++i) {  // Skip floating base (first 6 in velocity space)
        double torque_limit = 100.0;  // Default: 100 Nm
        
        // Use URDF effort limit if available and reasonable
        if (i < model_.effortLimit.size() && model_.effortLimit[i] < 1e10 && model_.effortLimit[i] > 0) {
            torque_limit = model_.effortLimit[i];
        }
        
        ::casadi::SX torque_upper_viol = ::casadi::SX::fmax(0, u(i) - torque_limit);
        ::casadi::SX torque_lower_viol = ::casadi::SX::fmax(0, -torque_limit - u(i));
        penalty += weights.penalty_joint * (torque_upper_viol * torque_upper_viol + 
                                           torque_lower_viol * torque_lower_viol);
    }
    
    // Ground contact constraints
    for (const auto& ee_name : ee_names_) {
        ::casadi::SX foot_pos = fkFramePos(q, ee_name);
        ::casadi::SX penetration = ::casadi::SX::fmax(0, -foot_pos(2));
        penalty += weights.penalty_contact * (penetration * penetration);
    }
    
    return penalty;
}

// Helper function implementations
::casadi::SX RobotUtils::fkFramePos(const ::casadi::SX& q, const std::string& frame_name) const {
    typedef ::casadi::SX Scalar;
    typedef pinocchio::ModelTpl<Scalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVector;
    
    ADModel ad_model = model_.cast<Scalar>();
    ADData ad_data(ad_model);
    
    ConfigVector q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q(i);
    }
    
    // Find frame ID
    FrameIndex frame_id = model_.getFrameId(frame_name);
    if (frame_id >= model_.frames.size()) {
        throw std::runtime_error("Frame '" + frame_name + "' not found");
    }
    
    // Compute kinematics
    pinocchio::forwardKinematics(ad_model, ad_data, q_ad);
    pinocchio::updateFramePlacements(ad_model, ad_data);
    
    // Extract position
    const auto& frame_placement = ad_data.oMf[frame_id];
    ::casadi::SX pos = ::casadi::SX::zeros(3, 1);
    
    for (int i = 0; i < 3; ++i) {
        pos(i) = frame_placement.translation()(i);
    }
    
    return pos;
}

::casadi::SX RobotUtils::jacFramePos(const ::casadi::SX& q, const std::string& frame_name) const {
    typedef ::casadi::SX Scalar;
    typedef pinocchio::ModelTpl<Scalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVector;
    
    ADModel ad_model = model_.cast<Scalar>();
    ADData ad_data(ad_model);
    
    ConfigVector q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q(i);
    }
    
    FrameIndex frame_id = model_.getFrameId(frame_name);
    if (frame_id >= model_.frames.size()) {
        throw std::runtime_error("Frame '" + frame_name + "' not found");
    }
    
    // Create the output Jacobian matrix (6 x nv) and initialize to zero
    Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J_eigen(6, nv_);
    J_eigen.setZero();
    
    // Compute Jacobian with the correct function signature
    pinocchio::computeFrameJacobian(ad_model, ad_data, q_ad, frame_id, pinocchio::LOCAL, J_eigen);
    
    // Extract position part (first 3 rows) and convert to ::casadi SX
    ::casadi::SX jac = ::casadi::SX::zeros(3, nv_);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < nv_; ++j) {
            jac(i, j) = J_eigen(i, j);
        }
    }
    
    return jac;
}


::casadi::SX RobotUtils::centerOfMass(const ::casadi::SX& q) const {
    typedef ::casadi::SX Scalar;
    typedef pinocchio::ModelTpl<Scalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVector;
    
    ADModel ad_model = model_.cast<Scalar>();
    ADData ad_data(ad_model);
    
    ConfigVector q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q(i);
    }
    
    // Compute center of mass
    pinocchio::centerOfMass(ad_model, ad_data, q_ad);
    
    ::casadi::SX com = ::casadi::SX::zeros(3, 1);
    for (int i = 0; i < 3; ++i) {
        com(i) = ad_data.com[0](i);
    }
    
    return com;
}

::casadi::SX RobotUtils::angularMomentum(const ::casadi::SX& q, const ::casadi::SX& qd) const {
    typedef ::casadi::SX Scalar;
    typedef pinocchio::ModelTpl<Scalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVector;
    typedef typename ADModel::TangentVectorType TangentVector;
    
    ADModel ad_model = model_.cast<Scalar>();
    ADData ad_data(ad_model);
    
    ConfigVector q_ad(nq_);
    TangentVector qd_ad(nv_);
    
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q(i);
    }
    for (int i = 0; i < nv_; ++i) {
        qd_ad(i) = qd(i);
    }
    
    // Compute centroidal momentum
    ccrba(ad_model, ad_data, q_ad, qd_ad);
    
    // Extract angular momentum
    ::casadi::SX L = ::casadi::SX::zeros(3, 1);
    for (int i = 0; i < 3; ++i) {
        L(i) = ad_data.hg.angular()(i);
    }
    
    return L;
}

::casadi::SX RobotUtils::symInverseDynamics(const ::casadi::SX& q, const ::casadi::SX& qd, const ::casadi::SX& qdd) const {
    // Create function only once and cache it
    if (!inv_dyn_func_initialized_) {
        typedef ::casadi::SX Scalar;
        typedef pinocchio::ModelTpl<Scalar> ADModel;
        typedef typename ADModel::Data ADData;
        typedef typename ADModel::ConfigVectorType ConfigVector;
        typedef typename ADModel::TangentVectorType TangentVector;
        
        ADModel ad_model = model_.cast<Scalar>();
        ADData ad_data(ad_model);
        
        // Create symbolic variables
        ::casadi::SX q_sym = ::casadi::SX::sym("q_inv", nq_);
        ::casadi::SX qd_sym = ::casadi::SX::sym("qd_inv", nv_);
        ::casadi::SX qdd_sym = ::casadi::SX::sym("qdd_inv", nv_);
        
        ConfigVector q_ad(nq_);
        TangentVector qd_ad(nv_);
        TangentVector qdd_ad(nv_);
        
        for (int i = 0; i < nq_; ++i) {
            q_ad(i) = q_sym(i);
        }
        for (int i = 0; i < nv_; ++i) {
            qd_ad(i) = qd_sym(i);
            qdd_ad(i) = qdd_sym(i);
        }
        
        // Compute inverse dynamics: tau = M(q)*qdd + h(q,qd)
        pinocchio::rnea(ad_model, ad_data, q_ad, qd_ad, qdd_ad);
        
        ::casadi::SX tau = ::casadi::SX::zeros(nv_, 1);
        for (int i = 0; i < nv_; ++i) {
            tau(i) = ad_data.tau(i);
        }
        
        // Cache the function
        inverse_dynamics_func_ = ::casadi::Function("inverse_dynamics", {q_sym, qd_sym, qdd_sym}, {tau});
        inv_dyn_func_initialized_ = true;
    }
    
    // Use cached function
    return inverse_dynamics_func_(std::vector<::casadi::SX>{q, qd, qdd})[0];
}

::casadi::SX RobotUtils::computeSoftContactForces(const ::casadi::SX& q, const ::casadi::SX& qd) const {
    ::casadi::SX contact_forces = ::casadi::SX::zeros(6 * ee_names_.size(), 1);
    
    for (size_t i = 0; i < ee_names_.size(); ++i) {
        // Get actual foot position using your existing fkFramePos function
        ::casadi::SX foot_pos = fkFramePos(q, ee_names_[i]);
        
        // Soft contact model
        double k_contact = 1e5;  // Contact stiffness
        double d_contact = 1e3;  // Contact damping
        
        // Normal force computation
        ::casadi::SX penetration = ::casadi::SX::fmax(0, -foot_pos(2)); // Ground at z=0
        ::casadi::SX normal_force = k_contact * penetration;
        normal_force = ::casadi::SX::fmax(0, normal_force);
        
        // Store normal force (fz component)
        contact_forces(i*6 + 2) = normal_force;
    }
    
    return contact_forces;
}

::casadi::SX RobotUtils::computeContactJacobians(const ::casadi::SX& q) const {
    std::vector<::casadi::SX> jacobian_blocks;
    
    for (size_t i = 0; i < ee_names_.size(); ++i) {
        // Get 3×nv position Jacobian using your existing function
        ::casadi::SX J_pos = jacFramePos(q, ee_names_[i]);
        
        // Create 6×nv Jacobian (position + zero orientation)
        ::casadi::SX J_contact = ::casadi::SX::vertcat({
            J_pos,                           // Position part
            ::casadi::SX::zeros(3, nv_)       // Orientation part (simplified)
        });
        
        jacobian_blocks.push_back(J_contact);
    }
    
    return ::casadi::SX::vertcat(jacobian_blocks);
}

::casadi::DM RobotUtils::computeContactForcesNumeric(const std::vector<::casadi::DM>& x_traj,
                                                  const std::vector<::casadi::DM>& u_traj) const {
    std::vector<::casadi::DM> contact_forces_traj;
    
    // Create cached functions for efficiency
    ::casadi::SX q_sym = ::casadi::SX::sym("q", nq_);
    ::casadi::SX qd_sym = ::casadi::SX::sym("qd", nv_);
    
    ::casadi::SX contact_forces_expr = computeSoftContactForces(q_sym, qd_sym);
    ::casadi::Function contact_func = ::casadi::Function("contact_forces", {q_sym, qd_sym}, {contact_forces_expr});
    
    for (size_t k = 0; k < x_traj.size() - 1; ++k) {
        ::casadi::DM q_k = x_traj[k](::casadi::Slice(0, nq_));
        ::casadi::DM qd_k = x_traj[k](::casadi::Slice(nq_, nx_));
        
        ::casadi::DM forces_k = contact_func(std::vector<::casadi::DM>{q_k, qd_k})[0];
        contact_forces_traj.push_back(forces_k);
    }
    
    return ::casadi::DM::vertcat(contact_forces_traj);
}

// Add this new function to robot_utils.cpp
::casadi::SX RobotUtils::computeConfigurationDerivative(const ::casadi::SX& q, const ::casadi::SX& qd) const {
    ::casadi::SX qdot = ::casadi::SX::zeros(nq_, 1);
    
    // Base position derivative (straightforward)
    qdot(::casadi::Slice(0, 3)) = qd(::casadi::Slice(0, 3));
    
    // Quaternion derivative (this is the critical part!)
    ::casadi::SX quat = q(::casadi::Slice(3, 7));
    ::casadi::SX omega = qd(::casadi::Slice(3, 6));
    ::casadi::SX quat_dot = computeQuaternionDerivative(quat, omega);
    qdot(::casadi::Slice(3, 7)) = quat_dot;
    
    // Joint derivatives (straightforward)
    qdot(::casadi::Slice(7, nq_)) = qd(::casadi::Slice(6, nv_));
    
    return qdot;
}

::casadi::SX RobotUtils::computeQuaternionDerivative(const ::casadi::SX& quat, const ::casadi::SX& omega) const {
    // Quaternion derivative: q̇ = 0.5 * Q(q) * [0; ω]
    // where Q(q) is the quaternion multiplication matrix
    
    ::casadi::SX qw = quat(0), qx = quat(1), qy = quat(2), qz = quat(3);
    ::casadi::SX wx = omega(0), wy = omega(1), wz = omega(2);
    
    // Quaternion derivative formula
    ::casadi::SX qw_dot = -0.5 * (qx * wx + qy * wy + qz * wz);
    ::casadi::SX qx_dot =  0.5 * (qw * wx + qy * wz - qz * wy);
    ::casadi::SX qy_dot =  0.5 * (qw * wy + qz * wx - qx * wz);
    ::casadi::SX qz_dot =  0.5 * (qw * wz + qx * wy - qy * wx);
    
    return ::casadi::SX::vertcat({qw_dot, qx_dot, qy_dot, qz_dot});
}

