#include "robot_model.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <stdexcept>


RobotModel::RobotModel(const std::string& urdf_path, const std::vector<std::string>& ee_names)
    : ee_names_(ee_names)
{
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);

    lower_limits_.resize(model_.nq);
    upper_limits_.resize(model_.nq);
    for (int i = 0; i < model_.nq; ++i) {
        lower_limits_[i] = model_.lowerPositionLimit[i];
        upper_limits_[i] = model_.upperPositionLimit[i];
    }

    // Build end-effector name to frame index map
    for (const auto& name : ee_names_) {
        ee_name_to_frame_[name] = model_.getFrameId(name);
    }
}

int RobotModel::getNumJoints() const {
    return model_.nq;
}

std::vector<double> RobotModel::getLowerLimits() const {
    return lower_limits_;
}

std::vector<double> RobotModel::getUpperLimits() const {
    return upper_limits_;
}

void RobotModel::computeForwardKinematics(const std::vector<double>& q) {
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    pinocchio::forwardKinematics(model_, data_, q_eigen);
    pinocchio::updateFramePlacements(model_, data_);
}

Eigen::Vector3d RobotModel::getEndEffectorPosition(const std::string& ee_name) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    return data_.oMf[it->second].translation();
}

Eigen::Matrix3d RobotModel::getEndEffectorOrientation(const std::string& ee_name) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    return data_.oMf[it->second].rotation();
}

Eigen::Quaterniond RobotModel::getEndEffectorQuaternion(const std::string& ee_name) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    return Eigen::Quaterniond(data_.oMf[it->second].rotation());
}

Eigen::MatrixXd RobotModel::getJacobian(const std::string& ee_name, const std::vector<double>& q) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::MatrixXd J(6, model_.nv);
    J.setZero();

    pinocchio::computeJointJacobians(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    pinocchio::updateFramePlacements(model_, const_cast<pinocchio::Data&>(data_));
    pinocchio::getFrameJacobian(model_, const_cast<pinocchio::Data&>(data_), it->second, pinocchio::LOCAL_WORLD_ALIGNED, J);

    return J;
}

Eigen::Vector3d RobotModel::getEndEffectorLinearVelocity(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::VectorXd v_eigen = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());

    pinocchio::computeJointJacobians(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    pinocchio::updateFramePlacements(model_, const_cast<pinocchio::Data&>(data_));
    pinocchio::forwardKinematics(model_, const_cast<pinocchio::Data&>(data_), q_eigen, v_eigen);

    pinocchio::Motion motion = pinocchio::getFrameVelocity(model_, const_cast<pinocchio::Data&>(data_), it->second, pinocchio::LOCAL_WORLD_ALIGNED);
    return motion.linear();
}

Eigen::Vector3d RobotModel::getEndEffectorAngularVelocity(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::VectorXd v_eigen = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());

    pinocchio::computeJointJacobians(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    pinocchio::forwardKinematics(model_, const_cast<pinocchio::Data&>(data_), q_eigen, v_eigen);
    pinocchio::updateFramePlacements(model_, const_cast<pinocchio::Data&>(data_));
    pinocchio::Motion motion = pinocchio::getFrameVelocity(model_, data_, it->second, pinocchio::LOCAL_WORLD_ALIGNED);
    return motion.angular();
}

Eigen::Vector3d RobotModel::getEndEffectorLinearAcceleration(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v, const std::vector<double>& a) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::VectorXd v_eigen = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    Eigen::VectorXd a_eigen = Eigen::Map<const Eigen::VectorXd>(a.data(), a.size());

    pinocchio::computeJointJacobians(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    pinocchio::forwardKinematics(model_, const_cast<pinocchio::Data&>(data_), q_eigen, v_eigen, a_eigen);
    pinocchio::updateFramePlacements(model_, const_cast<pinocchio::Data&>(data_));
    pinocchio::Motion motion = pinocchio::getFrameAcceleration(model_, data_, it->second, pinocchio::LOCAL_WORLD_ALIGNED);
    return motion.linear();
}

Eigen::Vector3d RobotModel::getEndEffectorAngularAcceleration(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v, const std::vector<double>& a) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::VectorXd v_eigen = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    Eigen::VectorXd a_eigen = Eigen::Map<const Eigen::VectorXd>(a.data(), a.size());

    pinocchio::computeJointJacobians(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    pinocchio::forwardKinematics(model_, const_cast<pinocchio::Data&>(data_), q_eigen, v_eigen, a_eigen);
    pinocchio::updateFramePlacements(model_, const_cast<pinocchio::Data&>(data_));
    pinocchio::Motion motion = pinocchio::getFrameAcceleration(model_, data_, it->second, pinocchio::LOCAL_WORLD_ALIGNED);
    return motion.angular();
}

double RobotModel::quaternionDistance(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) const {
    double dot = std::abs(q1.dot(q2));
    dot = std::min(1.0, std::max(-1.0, dot));
    return 2.0 * std::acos(dot);
}

Eigen::MatrixXd RobotModel::getMassMatrix(const std::vector<double>& q) const {
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    pinocchio::crba(model_, const_cast<pinocchio::Data&>(data_), q_eigen);
    return data_.M;
}

Eigen::VectorXd RobotModel::getBiasTerms(const std::vector<double>& q, const std::vector<double>& v) const {
    Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    Eigen::VectorXd v_eigen = Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    return pinocchio::nonLinearEffects(model_, const_cast<pinocchio::Data&>(data_), q_eigen, v_eigen);
}

pinocchio::FrameIndex RobotModel::getFrameId(const std::string& ee_name) const {
    auto it = ee_name_to_frame_.find(ee_name);
    if (it == ee_name_to_frame_.end()) throw std::runtime_error("Unknown EE name");
    return it->second;
}
