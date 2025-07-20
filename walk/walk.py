import pinocchio
import numpy as np
import crocoddyl
import os
from robot_gait_generator.crocoddyl import CrocoddylGaitProblemsInterface
import csv
import pinocchio.visualize
from scipy.spatial.transform import Rotation as R

os.environ["ROS_PACKAGE_PATH"] = "/home/prem/mpc/robot:" + os.environ.get("ROS_PACKAGE_PATH", "")
# Path to your URDF
urdf_path = "/home/prem/mpc/robot/h1_description/urdf/h1.urdf"

# Load the robot model
model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())

print(f"model.nq: {model.nq}")
print(f"model.nv: {model.nv}")
# Initial state (neutral pose)
q0 = pinocchio.neutral(model)
x0 = np.concatenate([q0, np.zeros(model.nv)])

# Foot link names as in your URDF
LEG_ORDER = ["left_ankle_link", "right_ankle_link"] 

# Gait parameters for bipedal walking
GAIT_PARAMETERS = {
    "step_frequencies": [1, 1],
    "duty_cycles": [0.7, 0.7],
    "phase_offsets": [0.0, 0.5],
    "relative_feet_targets": [[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]],  # stride length for each foot
    "foot_lift_height": [0.05, 0.05],
}

# Instantiate the gait problem interface
gait_problem_interface = CrocoddylGaitProblemsInterface(
    pinocchio_robot_model=model,
    default_standing_configuration=q0,
    ee_names=LEG_ORDER,
)

# Create the Crocoddyl shooting problem for walking gait
problem = gait_problem_interface.create_generic_gait_problem(
    x0=x0,
    starting_feet_heights=[0.0, 0.0],
    duration=10.0,                       #Duration
    time_step=0.05,                     #Time Step
    **GAIT_PARAMETERS
)

print(f"created problem successfully for time steps {len(problem.runningModels)}")

# Solve for the trajectory using DDP
solver = crocoddyl.SolverFDDP(problem)
solver.solve([], [], 100)

# Extract reference trajectories
reference_trajectory = solver.xs
reference_controls = solver.us

print(f"REF TRAJECTORY LENGTH: {len(reference_trajectory[0])}")

# Duration and time step
dt = 0.05
num_steps = len(reference_trajectory) - 1  # N steps, N+1 states
step_frequencies = [2.0, 2.0]
duty_cycles = [0.7, 0.7]
phase_offsets = [0.0, 0.5]

# Generate contact schedule
contact_schedule = []
for t in range(num_steps):
    time = t * dt
    contacts = []
    for freq, duty, phase_offset in zip(step_frequencies, duty_cycles, phase_offsets):
        phase = ((freq * time + phase_offset) % 1.0)
        in_contact = phase < duty
        contacts.append(in_contact)
    contact_schedule.append(contacts)

print(f"Generated {len(contact_schedule)} contact schedule steps")

# VISUALIZE
mesh_dir = "/home/prem/mpc/robot/h1_description/meshes"
# Load visual model for visualization
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path, mesh_dir, pinocchio.JointModelFreeFlyer())

# viz = pinocchio.visualize.MeshcatVisualizer(model, collision_model, visual_model)

# # Initialize the viewer
# viz.initViewer(open=True)
# viz.loadViewerModel()

# # Animate the trajectory
# import time

# for x in reference_trajectory:
#     viz.display(x[:model.nq])
#     time.sleep(0.02)  # Adjust for smoother/faster animation


# Create data object for computations
data = model.createData()

# Extract trajectories
q_ref = np.array([x[:model.nq] for x in reference_trajectory])
v_ref = np.array([x[model.nq:] for x in reference_trajectory])
q_ref[:, 0] = -q_ref[:, 0]
q_ref[:, 2] = q_ref[:, 2] + 0.14
q_ref[:, 1] = -q_ref[:, 1] 
q_trajecotry = q_ref[30:]
v_trajectory = v_ref[30:]

# for i in range(100): 
#     print(f"tau shape: {len(reference_controls[i])}")

# tau_ref = np.array([u[:] for u in reference_controls])
# np.savetxt("tau_ref.csv", tau_ref, delimiter=',')
# print(f"Saved tau_ref.csv with shape {tau_ref.shape}")

print(f"q_ref shape: {q_trajecotry.shape}")
print(f"v_ref shape: {v_trajectory.shape}")

# Save basic trajectories
np.savetxt("q_ref.csv", q_trajecotry, delimiter=',')
np.savetxt("v_ref.csv", v_trajectory, delimiter=',')

# Save contact schedule
contact_schedule_np = np.array(contact_schedule, dtype=int)
np.savetxt("contact_schedule.csv", contact_schedule_np, fmt='%d', delimiter=',')


# End-effector positions
ee_names = ["left_ankle_link", "right_ankle_link"]
ee_pos_ref = []

print("Computing end-effector positions...")
for i, q in enumerate(q_ref):
    # CRITICAL: Proper sequence for frame computation
    pinocchio.forwardKinematics(model, data, q)  # Compute joint placements first
    pinocchio.updateFramePlacements(model, data)  # Then update frame placements
    
    ee_positions = []
    for name in ee_names:
        try:
            frame_id = model.getFrameId(name)
            position = data.oMf[frame_id].translation  # Get position from frame placement
            ee_positions.extend(position)  # Flatten [x, y, z] for each EE
            
            if i == 0:  # Debug first iteration
                print(f"  {name}: {position}")
        except Exception as e:
            print(f"Error getting frame {name}: {e}")
            ee_positions.extend([0.0, 0.0, 0.0])  # Fallback
    
    ee_pos_ref.append(ee_positions)

ee_pos_ref = np.array(ee_pos_ref)
print(f"End-effector positions shape: {ee_pos_ref.shape}")
np.savetxt("ee_pos_ref.csv", ee_pos_ref, delimiter=',')

# ee_ori_ref = []
# print("Computing end-effector orientations...")
# for i, q in enumerate(q_ref):
#     pinocchio.forwardKinematics(model, data, q)
#     pinocchio.updateFramePlacements(model, data)
    
#     ee_orientations = []
#     for name in ee_names:
#         try:
#             frame_id = model.getFrameId(name)
#             rotation_matrix = data.oMf[frame_id].rotation
#             # CORRECTED: Use [x, y, z, w] convention for Pinocchio
#             quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [x, y, z, w]
#             ee_orientations.extend(quat)
            
#             if i == 0:  # Debug first iteration
#                 print(f"  {name} quaternion [x,y,z,w]: {quat}")
#         except Exception as e:
#             print(f"Error getting orientation for {name}: {e}")
#             ee_orientations.extend([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    
#     ee_ori_ref.append(ee_orientations)

# ee_ori_ref = np.array(ee_ori_ref)
# print(f"End-effector orientations shape: {ee_ori_ref.shape}")
# np.savetxt("ee_ori_ref.csv", ee_ori_ref, delimiter=',')

# # CORRECTED: Center of Mass computation
# print("Computing center of mass...")
# com_ref = []
# for q in q_ref:
#     pinocchio.centerOfMass(model, data, q)  # Compute CoM
#     com_ref.append(data.com[0].copy())  # Extract CoM position

# com_ref = np.array(com_ref)
# print(f"CoM reference shape: {com_ref.shape}")
# np.savetxt("com_ref.csv", com_ref, delimiter=',')

# # CORRECTED: ZMP computation using proper acceleration calculation
# print("Computing ZMP reference...")
# g = 9.81
# zmp_ref = []

# for k in range(len(a_ref)):
#     q = q_ref[k]
#     v = v_ref[k]
#     a = a_ref[k]
    
#     # Compute CoM and its derivatives
#     pinocchio.centerOfMass(model, data, q, v)  # Compute CoM position and velocity
#     pinocchio.jacobianCenterOfMass(model, data, q)  # Compute CoM Jacobian
    
#     com_pos = data.com[0].copy()
    
#     # CORRECTED: Compute CoM acceleration using Jacobian
#     com_ddot = np.dot(data.Jcom, a)  # CoM acceleration = J_com * joint_acceleration
    
#     # ZMP formula: ZMP = CoM_xy - (CoM_z / (CoM_ddot_z + g)) * CoM_ddot_xy
#     denom = com_ddot[2] + g
#     if abs(denom) < 1e-4:
#         denom = 1e-4  # Avoid division by zero
    
#     zmp_x = com_pos[0] - (com_pos[2] / denom) * com_ddot[0]
#     zmp_y = com_pos[1] - (com_pos[2] / denom) * com_ddot[1]
    
#     zmp_ref.append([zmp_x, zmp_y])

# zmp_ref = np.array(zmp_ref)
# print(f"ZMP reference shape: {zmp_ref.shape}")
# np.savetxt("zmp_ref.csv", zmp_ref, delimiter=',')
