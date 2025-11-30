# kalman_foundations

A comprehensive implementation of the Kalman Filter family exploring its mathematical foundations, from linear to nonlinear systems, including quaternion-based orientation estimation.

## Overview

This repository demonstrates:
1. **Linear Kalman Filter** - Four mathematical perspectives that all produce identical results
2. **Extended Kalman Filter (EKF)** - For nonlinear systems using linearization
3. **Unscented Kalman Filter (UKF)** - For nonlinear systems using sigma points
4. **Quaternion EKF/UKF** - For 3D orientation estimation (IMU/AHRS applications)

## Linear Kalman Filter - Four Mathematical Perspectives

The Linear Kalman Filter can be derived from four completely different mathematical perspectives, yet all four derivations converge to exactly the same numerical results.

### 1. Bayesian (`src/bayesian/`)
- **Perspective:** Probability Density Functions
- **Key Insight:** The update step is the product of two Gaussian distributions (Prior × Likelihood)
- **Terminology:** Uncertainty Propagation, Bayes' Rule, Posterior Distribution

### 2. Geometric (`src/geometric/`)
- **Perspective:** Orthogonal Projection in Hilbert Spaces
- **Key Insight:** The optimal estimate is the projection of the state onto the observation space
- **Terminology:** Hilbert Space, Orthogonality Principle, Projection Operator

### 3. Statistical (`src/statistical/`)
- **Perspective:** Minimum Mean Squared Error (MMSE) Estimation
- **Key Insight:** The update step solves a Weighted Least Squares optimization problem
- **Terminology:** Cost Function J(x), WLS, MMSE Estimator

### 4. Optimal Observer (`src/optimal_observer/`)
- **Perspective:** Control Theory and State Observer Design
- **Key Insight:** The Kalman gain is the optimal observer gain from the Riccati equation
- **Terminology:** Error Dynamics, Feedback Gain K, Stability, Riccati Equation

## Nonlinear Kalman Filters

### Extended Kalman Filter (`src/ekf/`)
The EKF handles nonlinear systems by linearizing the dynamics and measurement functions using Jacobians.

**Key Features:**
- First-order Taylor series approximation
- Requires analytical Jacobian computation
- Computationally efficient
- Best for mildly nonlinear systems

**Example:** Nonlinear pendulum tracking with damping

### Unscented Kalman Filter (`src/ukf/`)
The UKF uses sigma points to capture the probability distribution without linearization.

**Key Features:**
- No Jacobian computation required
- Captures mean and covariance to 3rd order accuracy
- Better for highly nonlinear systems
- Slightly more computationally expensive than EKF

**Example:** Same pendulum problem - compare accuracy vs EKF

## Quaternion-Based Orientation Estimation

### Quaternion EKF (`src/quaternion_ekf/`)
Extended Kalman Filter for 3D orientation estimation using quaternions.

**Key Features:**
- Avoids gimbal lock (unlike Euler angles)
- Proper quaternion normalization constraint handling
- Fuses gyroscope, accelerometer, and magnetometer
- Used in AHRS (Attitude and Heading Reference Systems)

### Quaternion UKF (`src/quaternion_ukf/`)
Unscented Kalman Filter for 3D orientation with proper quaternion manifold handling.

**Key Features:**
- Sigma points generated on quaternion manifold
- Quaternion mean via iterative averaging
- Error-state formulation for numerical stability
- Superior accuracy for aggressive maneuvers

**Applications:**
- IMU sensor fusion
- Drone/robot orientation tracking
- Spacecraft attitude estimation
- VR/AR head tracking

## Requirements

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.14 or higher
- **Eigen3** linear algebra library

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake libeigen3-dev
```

## Building

```bash
# Clone the repository
git clone https://github.com/your-username/kalman_foundations.git
cd kalman_foundations

# Build
mkdir build && cd build
cmake ..
make
```

## Running the Demos

### 1. Linear Kalman Filter Demo
Runs a 1D constant velocity tracking simulation through all four mathematical perspectives:
```bash
./build/kalman_demo
```
Expected output shows all four implementations produce **identical floating-point results**.

### 2. EKF vs UKF Demo
Compares EKF and UKF on a nonlinear pendulum tracking problem:
```bash
./build/ekf_ukf_demo
```
Both filters track the nonlinear system effectively.

### 3. Quaternion EKF/UKF Demo
Simulates 3D orientation estimation using IMU sensor fusion:
```bash
./build/quaternion_demo
```
Demonstrates quaternion-based AHRS with gyroscope, accelerometer, and magnetometer fusion.

## Project Structure

```
kalman_foundations/
├── CMakeLists.txt                    # Build configuration
├── README.md                         # This file
├── .gitignore                        # Git ignore patterns
└── src/
    ├── main.cpp                      # Linear KF demo
    ├── ekf_ukf_demo.cpp              # EKF vs UKF comparison demo
    ├── quaternion_demo.cpp           # Quaternion filter demo
    ├── bayesian/
    │   └── KalmanFilter.hpp          # Bayesian perspective
    ├── geometric/
    │   └── KalmanFilter.hpp          # Geometric perspective
    ├── statistical/
    │   └── KalmanFilter.hpp          # Statistical perspective
    ├── optimal_observer/
    │   └── KalmanFilter.hpp          # Control theory perspective
    ├── ekf/
    │   └── ExtendedKalmanFilter.hpp  # Extended Kalman Filter
    ├── ukf/
    │   └── UnscentedKalmanFilter.hpp # Unscented Kalman Filter
    ├── quaternion_ekf/
    │   └── QuaternionEKF.hpp         # Quaternion EKF
    └── quaternion_ukf/
        └── QuaternionUKF.hpp         # Quaternion UKF
```

## Mathematical Details

### The Pendulum Model (EKF/UKF)

**State vector:** `x = [θ, θ̇]ᵀ` (angle and angular velocity)

**Nonlinear Dynamics:**
```
θ̈ = -(g/L)·sin(θ) - b·θ̇
```
where g=9.81 m/s², L=1.0 m, b=0.1 (damping)

### The Quaternion Model (3D Orientation)

**State vector:** `x = [qw, qx, qy, qz, ωx, ωy, ωz]ᵀ`

**Quaternion Kinematics:**
```
q̇ = 0.5 · q ⊗ [0, ω]
```
where ⊗ is quaternion multiplication

**Measurements:**
- Accelerometer: Gravity vector in body frame → Roll/Pitch correction
- Magnetometer: Magnetic field in body frame → Yaw/Heading correction
- Gyroscope: Angular velocity → Orientation integration

## License

MIT License
