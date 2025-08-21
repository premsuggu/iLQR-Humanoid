# iLQR-Humanoid: Model Predictive Control for Humanoid Robots

A sophisticated implementation of iterative Linear Quadratic Regulator (iLQR) with First-Order Hold (FOH) dynamics for humanoid robot control. This project implements Model Predictive Control (MPC) using advanced optimal control techniques for bipedal locomotion.

## Features

- **FOH-iLQR**: First-Order Hold implementation of iLQR for improved trajectory smoothness
- **Simpson's Rule Integration**: High-accuracy cost function evaluation using Simpson's quadrature
- **Constraint Handling**: Augmented Lagrangian method for joint limits and contact constraints
- **Real-time MPC**: Efficient symbolic differentiation using CasADi for real-time performance
- **Humanoid Dynamics**: Full floating-base dynamics with contact force modeling
- **Pinocchio Integration**: Leverages Pinocchio for efficient rigid body dynamics

## Platform Requirements

⚠️ **Linux Only**: This project requires Linux due to Pinocchio dependency limitations. If you're on Windows, please use WSL (Windows Subsystem for Linux) or a Linux virtual machine.

**Tested on:**
- Ubuntu 20.04/22.04
- WSL2 with Ubuntu

## Dependencies

- **CasADi**: Symbolic framework for automatic differentiation and optimization
- **Pinocchio**: Fast and flexible implementation of rigid body dynamics algorithms
- **Eigen3**: Linear algebra library
- **CMake**: Build system (≥ 3.16)
- **C++17** compatible compiler

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/premsuggu/iLQR-Humanoid.git
cd iLQR-Humanoid
```

### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git pkg-config
sudo apt install libeigen3-dev libboost-all-dev
sudo apt install libxml2-dev libxslt-dev
sudo apt install python3-dev python3-numpy

# For URDF support
sudo apt install liburdfdom-dev liburdfdom-headers-dev
sudo apt install libassimp-dev liboctomap-dev libfcl-dev
```

### 3. Build CasADi from Source

```bash
# Clone CasADi
git clone https://github.com/casadi/casadi.git
cd casadi

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/mpc/install \
  -DWITH_PYTHON=OFF \
  -DWITH_MATLAB=OFF \
  -DWITH_OCTAVE=OFF

# Build and install (this may take 15-30 minutes)
make -j$(nproc)
make install

# Return to project root
cd ../../
```

### 4. Build Pinocchio from Source

```bash
# Clone Pinocchio with submodules
git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio

# Create build directory
mkdir -p build && cd build

# Configure with CasADi support
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/mpc/install \
  -DBUILD_PYTHON_INTERFACE=OFF \
  -DBUILD_WITH_CASADI_SUPPORT=ON \
  -DBUILD_WITH_URDF_SUPPORT=ON \
  -DBUILD_WITH_HPP_FCL=ON \
  -DBUILD_TESTING=OFF \

# Build and install (this may take 10-20 minutes)
make -j$(nproc)
make install

# Return to project root
cd ../../
```

### 5. Build the MPC Project

```bash
# Create build directory in project root
mkdir -p build && cd build

# Configure the project
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=~/mpc/install

# Build the project
make -j$(nproc)
```

## Running the Simulation

### 1. Prepare Reference Trajectory

Ensure you have the reference trajectory file:
```bash
# The project expects reference data at:
# ~/mpc/walk/q_ref.csv
ls ../walk/q_ref.csv  # Should exist
```

### 2. Run the Humanoid MPC

```bash
# From the build directory
./humanoid
```

### 3. Expected Output

The simulation will:
1. Load the H1 humanoid model from URDF
2. Initialize the iLQR solver with FOH dynamics
3. Load reference trajectory from CSV
4. Run MPC control loop for 500 time steps (10 seconds at 50Hz)
5. Save optimal trajectory to `q_optimal.csv`

## File Structure

```
├── README.md              # This file
├── CMakeLists.txt         # Build configuration
├── ilqr.hpp/cpp           # Main iLQR implementation
├── robot_utils.hpp/cpp    # Robot dynamics and utilities
├── mpc.cpp                # Main MPC application
├── test.cpp               # Unit tests
├── build/                 # Build directory (created)
├── robot/                 # Robot URDF files
│   └── h1_description/
├── walk/                  # Reference trajectories
│   ├── q_ref.csv         # Joint position references
│   ├── v_ref.csv         # Velocity references
│   └── walk.py           # Trajectory generation
└── install/              # Installation directory (created)
```

## Configuration Parameters

Key parameters can be modified in `mpc.cpp`:

```cpp
// MPC parameters
double dt = 0.02;       // Control frequency (50 Hz)
int N = 10;             // Horizon length (0.2 seconds)
int sim_steps = 500;    // Total simulation steps (10 seconds)

// Cost weights
Q(i, i) = 100.0;  // Base position tracking
Q(i, i) = 50.0;   // Base orientation tracking  
Q(i, i) = 10.0;   // Joint position tracking
Q(i, i) = 1.0;    // Velocity tracking
R(i, i) = 0.01;   // Control effort
```

## References

This implementation is based on:
- "First-Order Hold iLQR for Trajectory Optimization" 
- Simpson's rule integration for cost functions
- Augmented Lagrangian methods for constraints
- Pinocchio rigid body dynamics library
