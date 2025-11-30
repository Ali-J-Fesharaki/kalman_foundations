/**
 * @file main.cpp
 * @brief Demonstrates that all four Kalman Filter derivations produce identical results
 *
 * This program implements a 1D constant velocity tracking problem and runs it
 * through all four mathematical formulations of the Kalman Filter:
 *
 * 1. Bayesian (Probability Density Functions)
 * 2. Geometric (Orthogonal Projection)
 * 3. Statistical (Minimum Mean Squared Error)
 * 4. Optimal Observer (Control Theory)
 *
 * The key insight is that despite their different mathematical origins,
 * all four approaches produce EXACTLY the same numerical results.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

// Include all four implementations
#include "bayesian/KalmanFilter.hpp"
#include "geometric/KalmanFilter.hpp"
#include "statistical/KalmanFilter.hpp"
#include "optimal_observer/KalmanFilter.hpp"

/**
 * @brief Checks if two vectors are numerically equal within tolerance
 * @param a First vector
 * @param b Second vector
 * @param tolerance Maximum allowed difference per element
 * @return true if vectors are equal within tolerance
 */
bool vectorsEqual(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tolerance = 1e-12)
{
    if (a.size() != b.size()) return false;
    return (a - b).norm() < tolerance;
}

/**
 * @brief Checks if two matrices are numerically equal within tolerance
 * @param a First matrix
 * @param b Second matrix
 * @param tolerance Maximum allowed difference per element
 * @return true if matrices are equal within tolerance
 */
bool matricesEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tolerance = 1e-12)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    return (a - b).norm() < tolerance;
}

/**
 * @brief Prints a state vector with its label
 */
void printState(const std::string& label, const Eigen::VectorXd& x)
{
    std::cout << std::setw(20) << label << ": ["
              << std::fixed << std::setprecision(6)
              << x(0) << ", " << x(1) << "]" << std::endl;
}

int main()
{
    std::cout << "=========================================================\n";
    std::cout << "   Kalman Filter: Four Mathematical Derivations Demo\n";
    std::cout << "=========================================================\n\n";

    // =========================================================================
    // System Definition: 1D Constant Velocity Model
    // =========================================================================
    //
    // State vector: x = [position, velocity]'
    //
    // Dynamics: x[k+1] = F * x[k] + w[k]
    //   position[k+1] = position[k] + dt * velocity[k]
    //   velocity[k+1] = velocity[k]
    //
    // Measurement: z[k] = H * x[k] + v[k]
    //   We observe position only: z = position + noise

    const int state_dim = 2;  // [position, velocity]
    const int meas_dim = 1;   // [position measurement]

    double dt = 1.0;  // Time step

    // State transition matrix
    Eigen::MatrixXd F(state_dim, state_dim);
    F << 1, dt,
         0, 1;

    // Measurement matrix (we only observe position)
    Eigen::MatrixXd H(meas_dim, state_dim);
    H << 1, 0;

    // Process noise covariance (uncertainty in dynamics)
    double q = 0.1;  // Process noise intensity
    Eigen::MatrixXd Q(state_dim, state_dim);
    Q << (q * dt * dt * dt / 3), (q * dt * dt / 2),
         (q * dt * dt / 2),      (q * dt);

    // Measurement noise covariance
    double r = 1.0;  // Measurement noise variance
    Eigen::MatrixXd R(meas_dim, meas_dim);
    R << r;

    // Initial state estimate and covariance
    Eigen::VectorXd x0(state_dim);
    x0 << 0.0, 1.0;  // Start at position 0, velocity 1

    Eigen::MatrixXd P0(state_dim, state_dim);
    P0 << 1.0, 0.0,
          0.0, 1.0;

    // =========================================================================
    // Generate Simulated Measurements
    // =========================================================================

    // True initial state
    Eigen::VectorXd x_true(state_dim);
    x_true << 0.0, 1.0;  // Position 0, velocity 1 m/s

    // Random number generator for measurement noise
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> meas_noise(0.0, std::sqrt(r));

    // Generate measurements
    const int num_steps = 10;
    std::vector<double> measurements(num_steps);

    std::cout << "Ground Truth & Measurements:\n";
    std::cout << "----------------------------\n";
    std::cout << std::setw(6) << "Step" << std::setw(12) << "True Pos"
              << std::setw(12) << "Measured" << std::endl;

    Eigen::VectorXd x_current = x_true;
    for (int k = 0; k < num_steps; ++k)
    {
        // True state (deterministic for this demo)
        if (k > 0)
        {
            x_current = F * x_current;
        }

        // Measurement with noise
        double true_pos = x_current(0);
        double meas = true_pos + meas_noise(rng);
        measurements[k] = meas;

        std::cout << std::setw(6) << k
                  << std::setw(12) << std::fixed << std::setprecision(4) << true_pos
                  << std::setw(12) << meas << std::endl;
    }
    std::cout << std::endl;

    // =========================================================================
    // Initialize All Four Kalman Filters
    // =========================================================================

    bayesian::KalmanFilter kf_bayesian(state_dim, meas_dim);
    geometric::KalmanFilter kf_geometric(state_dim, meas_dim);
    statistical::KalmanFilter kf_statistical(state_dim, meas_dim);
    optimal_observer::KalmanFilter kf_observer(state_dim, meas_dim);

    // Configure all filters identically
    auto configureFilter = [&](auto& kf) {
        kf.setStateTransition(F);
        kf.setMeasurementMatrix(H);
        kf.setProcessNoise(Q);
        kf.setMeasurementNoise(R);
        kf.setState(x0);
        kf.setCovariance(P0);
    };

    configureFilter(kf_bayesian);
    configureFilter(kf_geometric);
    configureFilter(kf_statistical);
    configureFilter(kf_observer);

    // =========================================================================
    // Run All Four Filters and Compare Results
    // =========================================================================

    std::cout << "Running Kalman Filters...\n";
    std::cout << "=========================\n\n";

    bool all_match = true;

    for (int k = 0; k < num_steps; ++k)
    {
        std::cout << "--- Step " << k << " ---\n";

        // Create measurement vector
        Eigen::VectorXd z(meas_dim);
        z << measurements[k];

        // Predict step for all filters
        kf_bayesian.predict();
        kf_geometric.predict();
        kf_statistical.predict();
        kf_observer.predict();

        // Update step for all filters
        kf_bayesian.update(z);
        kf_geometric.update(z);
        kf_statistical.update(z);
        kf_observer.update(z);

        // Get states from all filters
        const auto& x_bay = kf_bayesian.getState();
        const auto& x_geo = kf_geometric.getState();
        const auto& x_stat = kf_statistical.getState();
        const auto& x_obs = kf_observer.getState();

        // Get covariances from all filters
        const auto& P_bay = kf_bayesian.getCovariance();
        const auto& P_geo = kf_geometric.getCovariance();
        const auto& P_stat = kf_statistical.getCovariance();
        const auto& P_obs = kf_observer.getCovariance();

        // Print states
        printState("Bayesian", x_bay);
        printState("Geometric", x_geo);
        printState("Statistical", x_stat);
        printState("Optimal Observer", x_obs);

        // Check if all states match
        bool states_match =
            vectorsEqual(x_bay, x_geo) &&
            vectorsEqual(x_bay, x_stat) &&
            vectorsEqual(x_bay, x_obs);

        bool covs_match =
            matricesEqual(P_bay, P_geo) &&
            matricesEqual(P_bay, P_stat) &&
            matricesEqual(P_bay, P_obs);

        if (states_match && covs_match)
        {
            std::cout << "✓ All four implementations produce IDENTICAL results!\n";
        }
        else
        {
            std::cout << "✗ MISMATCH DETECTED!\n";
            all_match = false;
        }

        std::cout << std::endl;
    }

    // =========================================================================
    // Final Summary
    // =========================================================================

    std::cout << "=========================================================\n";
    std::cout << "                     FINAL SUMMARY\n";
    std::cout << "=========================================================\n\n";

    std::cout << "Final state estimates:\n";
    printState("Bayesian", kf_bayesian.getState());
    printState("Geometric", kf_geometric.getState());
    printState("Statistical", kf_statistical.getState());
    printState("Optimal Observer", kf_observer.getState());
    std::cout << std::endl;

    std::cout << "Final covariance (Bayesian - representative):\n";
    std::cout << kf_bayesian.getCovariance() << std::endl << std::endl;

    if (all_match)
    {
        std::cout << "=========================================================\n";
        std::cout << "  SUCCESS: All four derivations produce identical results!\n";
        std::cout << "=========================================================\n\n";

        std::cout << "This demonstrates that whether you view the Kalman Filter as:\n";
        std::cout << "  1. Bayesian inference (product of Gaussians)\n";
        std::cout << "  2. Orthogonal projection in Hilbert space\n";
        std::cout << "  3. Minimum Mean Squared Error estimation\n";
        std::cout << "  4. Optimal state observer design\n";
        std::cout << "\nThe mathematical formulas are equivalent!\n";

        return 0;
    }
    else
    {
        std::cout << "=========================================================\n";
        std::cout << "  FAILURE: Results don't match - check implementation!\n";
        std::cout << "=========================================================\n";

        return 1;
    }
}
