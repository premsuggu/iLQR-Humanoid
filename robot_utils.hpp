#pragma once
#include <casadi/casadi.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/joint/fwd.hpp>

class RobotUtils {
public:
    explicit RobotUtils(const std::string& urdf_path);

    // Core symbolic expression generators
    ::casadi::SX symFloatingBaseDynamics(const ::casadi::SX& x, const ::casadi::SX& u) const;
    ::casadi::SX symCostStage(const ::casadi::SX& x, const ::casadi::SX& u,
                           const ::casadi::SX& x_ref, const ::casadi::SX& u_ref) const;
    ::casadi::SX symConstraintPenalty(const ::casadi::SX& x, const ::casadi::SX& u) const;
    ::casadi::SX symInverseDynamics(const ::casadi::SX& q, const ::casadi::SX& qd, const ::casadi::SX& qdd) const;
    ::casadi::SX computeSoftContactForces(const ::casadi::SX& q, const ::casadi::SX& qd) const;
    ::casadi::SX computeContactJacobians(const ::casadi::SX& q) const;
    ::casadi::DM computeContactForcesNumeric(const std::vector<::casadi::DM>& x_traj,
                                      const std::vector<::casadi::DM>& u_traj) const;

    // Robot parameters
    int nq() const { return nq_; }
    int nv() const { return nv_; }
    int nx() const { return nx_; }
    int nu() const { return nu_; }

    // End effector management
    const std::vector<std::string>& getEndEffectorNames() const { return ee_names_; }
    void setEndEffector(const std::vector<std::string>& ee_names) { ee_names_ = ee_names; }

    // Joint limits access
    const std::vector<double>& getLowerLimits() const { return lower_limits_; }
    const std::vector<double>& getUpperLimits() const { return upper_limits_; }
    const std::vector<bool>& getValidLimits() const { return valid_limits_; } 

    // Cost weights structure
    struct CostWeights {
        double Q_base_pos = 100.0;
        double Q_base_ori = 50.0;
        double Q_joints = 10.0;
        double Q_velocity = 1.0;
        double R_torque = 0.01;
        double R_smoothness = 0.10;
        double Q_ang_momentum = 10.0;
        double terminal_scale = 2.0;
        double penalty_joint = 5e3;
        double penalty_contact = 1e4;
        double penalty_friction = 1e3;
    } weights;

    void setCostWeights(const CostWeights& new_weights) { weights = new_weights; }
private:
    // Pinocchio model
    pinocchio::Model model_;
    pinocchio::Data data_;
    
    // Robot dimensions
    int nq_, nv_, nx_, nu_;

    // Joint limits and end effectors
    std::vector<double> lower_limits_;
    std::vector<double> upper_limits_;
    std::vector<bool> valid_limits_;     
    std::vector<std::string> ee_names_;

    mutable ::casadi::Function inverse_dynamics_func_;
    mutable bool inv_dyn_func_initialized_ = false;

    // Helper functions
    ::casadi::SX fkFramePos(const ::casadi::SX& q, const std::string& frame_name) const;
    ::casadi::SX jacFramePos(const ::casadi::SX& q, const std::string& frame_name) const;
    ::casadi::SX centerOfMass(const ::casadi::SX& q) const;
    ::casadi::SX angularMomentum(const ::casadi::SX& q, const ::casadi::SX& qd) const;
    ::casadi::SX computeConfigurationDerivative(const ::casadi::SX& q, const ::casadi::SX& qd) const;
    ::casadi::SX computeQuaternionDerivative(const ::casadi::SX& quat, const ::casadi::SX& omega) const;
};
