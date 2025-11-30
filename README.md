# kalman_foundations

A comprehensive implementation of the Kalman Filter exploring its four distinct mathematical origins: Bayesian probability, Geometric projection, Statistical estimation, and Control theoretic optimal observation.

## Overview

This repository demonstrates that the Linear Kalman Filter can be derived from four completely different mathematical perspectives, yet all four derivations converge to exactly the same numerical results. This unity across disparate fields of mathematics is one of the beautiful aspects of optimal estimation theory.

## Features

- **Four Complete Implementations**: Each perspective is implemented in its own namespace with detailed comments
- **Educational GUI**: Interactive visualization tool to help learn the Kalman Filter foundations
- **Console Demo**: Command-line demonstration showing all four implementations produce identical results
- **Real-time Plotting**: Visual comparison of estimates, measurements, and ground truth

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
- **GLFW3** (for GUI only)
- **OpenGL** (for GUI only)

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake libeigen3-dev libglfw3-dev
```

## Building

```bash
# Clone the repository
git clone https://github.com/your-username/kalman_foundations.git
cd kalman_foundations

# Build everything (console demo + GUI)
mkdir build && cd build
cmake ..
make

# Build console demo only (no GUI dependencies)
cmake -DBUILD_GUI=OFF ..
make
```

## Running

### Console Demo

The demo runs a 1D constant velocity tracking simulation through all four implementations:

```bash
./build/kalman_demo
```

Expected output will show that all four implementations produce **identical floating-point results** at each time step.

### Educational GUI

Launch the interactive learning tool:

```bash
./build/kalman_gui
```

The GUI provides:
- **Interactive Simulation**: Step through the filter manually or run automatically
- **Real-time Plots**: Position tracking, velocity estimation, and uncertainty visualization
- **Parameter Tuning**: Adjust process noise, measurement noise, and initial conditions
- **Educational Content**: Detailed explanations of each mathematical perspective
- **Implementation Comparison**: Live comparison showing all four implementations produce identical results

## Project Structure

```
kalman_foundations/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
├── extern/                     # Third-party libraries
│   ├── imgui/                  # Dear ImGui (GUI library)
│   └── implot/                 # ImPlot (plotting extension)
└── src/
    ├── main.cpp                # Console demo
    ├── gui/
    │   └── main_gui.cpp        # Educational GUI application
    ├── bayesian/
    │   └── KalmanFilter.hpp    # Bayesian perspective
    ├── geometric/
    │   └── KalmanFilter.hpp    # Geometric perspective
    ├── statistical/
    │   └── KalmanFilter.hpp    # Statistical perspective
    └── optimal_observer/
        └── KalmanFilter.hpp    # Control theory perspective
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

## Learning with the GUI

1. **Start**: Click "Initialize" to set up the simulation
2. **Step**: Click "Step" to advance one time step, or "Run Auto" for continuous simulation
3. **Observe**: Watch how:
   - The blue line (estimate) converges toward the green line (truth)
   - The orange dots (measurements) are noisy
   - The uncertainty decreases over time
4. **Experiment**: Adjust parameters to see their effects:
   - Increase measurement noise (R) → filter trusts measurements less
   - Increase process noise (Q) → filter expects more state variation
5. **Learn**: Read the "Mathematical Perspectives" tab to understand each derivation

## License

MIT License
