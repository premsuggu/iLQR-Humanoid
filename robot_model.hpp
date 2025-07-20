#ifndef ROBOT_MODEL_HPP
#define ROBOT_MODEL_HPP

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <unordered_map>


class RobotModel {
public:
    RobotModel(const std::string& urdf_path, const std::vector<std::string>& ee_names);

    int getNumJoints() const;
    std::vector<double> getLowerLimits() const;
    std::vector<double> getUpperLimits() const;

    // Kinematics
    void computeForwardKinematics(const std::vector<double>& q);
    Eigen::Vector3d getEndEffectorPosition(const std::string& ee_name) const;
    Eigen::Matrix3d getEndEffectorOrientation(const std::string& ee_name) const;
    Eigen::Quaterniond getEndEffectorQuaternion(const std::string& ee_name) const;
    Eigen::MatrixXd getJacobian(const std::string& ee_name, const std::vector<double>& q) const;
    Eigen::Vector3d getEndEffectorLinearVelocity(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v) const;
    Eigen::Vector3d getEndEffectorAngularVelocity(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v) const;
    Eigen::Vector3d getEndEffectorLinearAcceleration(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v, const std::vector<double>& a) const;
    Eigen::Vector3d getEndEffectorAngularAcceleration(const std::string& ee_name, const std::vector<double>& q, const std::vector<double>& v, const std::vector<double>& a) const;
    double quaternionDistance(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) const;

    // Dynamics
    Eigen::MatrixXd getMassMatrix(const std::vector<double>& q) const;
    Eigen::VectorXd getBiasTerms(const std::vector<double>& q, const std::vector<double>& v) const;

    // Frame index lookup
    pinocchio::FrameIndex getFrameId(const std::string& ee_name) const;

    // EE names
    const std::vector<std::string>& getEndEffectorNames() const { return ee_names_; }

    // Pinocchio model/data access (for CasADi integration)
    const pinocchio::Model& getModel() const { return model_; }
    pinocchio::Data& getData() { return data_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<double> lower_limits_;
    std::vector<double> upper_limits_;
    std::vector<std::string> ee_names_;
    std::unordered_map<std::string, pinocchio::FrameIndex> ee_name_to_frame_;
};

#endif // ROBOT_MODEL_HPP
