import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("/home/prem/wb_mpc/robots/h1_description/mjcf/scene.xml")
q_trajectory = pd.read_csv("/home/prem/mpc/walk/q_ref.csv", header=None).values
data = mujoco.MjData(model)

frame = 0
num_frames = len(q_trajectory)

# q_trajectory[:, 0] = -q_trajectory[:, 0]
q_trajectory[:, 2] = q_trajectory[:, 2] + 0.14
# q_trajectory[:, 1] = -q_trajectory[:, 1] 
q_trajectory[:, 2] = 1.043;
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Set qpos to current frame
        data.qpos[:] = q_trajectory[frame]
        mujoco.mj_forward(model, data)
        viewer.sync()
        frame = (frame + 1) % num_frames  # Loop or stop as needed
        time.sleep(1/60)  # ~60 FPS; adjust as needed

# tau_ref = pd.read_csv("/home/prem/mpc/walk/tau_ref.csv", header=None).values  # shape: (T, nv-6)
# q_init = q_trajectory[0]
# v_init = np.zeros(model.nv)

# data.qpos[:] = q_init
# data.qvel[:] = v_init
# mujoco.mj_forward(model, data)

# actual_tau = []
# for tau in tau_ref:
#     actual_tau.append(tau[6:])

# frame = 0
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("Press ESC to exit.")
#     while viewer.is_running() and frame < len(actual_tau):
#         data.ctrl[:] = actual_tau[frame]  # Apply joint torques only
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(1/60)  # Adjust as needed
#         frame += 1
