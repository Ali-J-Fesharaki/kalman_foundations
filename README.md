# kalman_foundations

A comprehensive implementation of the Kalman Filter exploring its four distinct mathematical origins: Bayesian probability, Geometric projection, Statistical estimation, and Control theoretic optimal observation.

## Overview

This repository demonstrates that the Linear Kalman Filter can be derived from four completely different mathematical perspectives, yet all four derivations converge to exactly the same numerical results. This unity across disparate fields of mathematics is one of the beautiful aspects of optimal estimation theory.

## Mathematical Perspectives

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

## Running the Demo

The demo runs a 1D constant velocity tracking simulation through all four implementations:

```bash
./build/kalman_demo
```

Expected output will show that all four implementations produce **identical floating-point results** at each time step.

## Project Structure

```
kalman_foundations/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
└── src/
    ├── main.cpp                # Demo: 1D constant velocity tracking
    ├── bayesian/
    │   └── KalmanFilter.hpp    # Bayesian perspective implementation
    ├── geometric/
    │   └── KalmanFilter.hpp    # Geometric perspective implementation
    ├── statistical/
    │   └── KalmanFilter.hpp    # Statistical perspective implementation
    └── optimal_observer/
        └── KalmanFilter.hpp    # Control theory perspective implementation
```

## The Constant Velocity Model

The demo implements a 1D constant velocity tracking problem:

**State vector:** `x = [position, velocity]ᵀ`

**Dynamics:**
```
x[k+1] = F · x[k] + w[k]
F = [1  dt]
    [0   1]
```

**Measurement:**
```
z[k] = H · x[k] + v[k]
H = [1  0]  (observe position only)
```

## License

MIT License
