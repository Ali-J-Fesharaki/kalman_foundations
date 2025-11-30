/**
 * @file ekf_ukf_demo.cpp
 * @brief Demonstrates EKF and UKF on a nonlinear pendulum system
 *
 * This demo compares the Extended Kalman Filter (EKF) and Unscented Kalman
 * Filter (UKF) on the same nonlinear pendulum tracking problem.
 *
 * System: Simple pendulum with damping
 * State: [θ, θ̇]' (angle and angular velocity)
 * Dynamics: θ̈ = -(g/L)*sin(θ) - b*θ̇
 * Measurement: We observe the angle θ with noise
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

#include "ekf/ExtendedKalmanFilter.hpp"

// Define PI for portability (not guaranteed in all compilers)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ukf/UnscentedKalmanFilter.hpp"

int main()
{
    std::cout << "=========================================================\n";
    std::cout << "     EKF vs UKF: Nonlinear Pendulum Tracking Demo\n";
    std::cout << "=========================================================\n\n";

    // =========================================================================
    // System Parameters
    // =========================================================================
    const double g = 9.81;   // Gravity (m/s²)
    const double L = 1.0;    // Pendulum length (m)
    const double b = 0.1;    // Damping coefficient
    const double dt = 0.01;  // Time step (s)

    const int state_dim = 2;
    const int meas_dim = 1;

    // =========================================================================
    // Create EKF
    // =========================================================================
    ekf::ExtendedKalmanFilter ekf_filter(state_dim, meas_dim);

    // Define nonlinear state transition using lambdas with captured parameters
    ekf_filter.setStateFunction([g, L, b](const Eigen::VectorXd& x, double dt_local) {
        return ekf::pendulum::stateTransition(x, dt_local, g, L, b);
    });

    ekf_filter.setStateJacobian([g, L, b](const Eigen::VectorXd& x, double dt_local) {
        return ekf::pendulum::stateJacobian(x, dt_local, g, L, b);
    });

    ekf_filter.setMeasurementFunction(ekf::pendulum::measurementFunction);
    ekf_filter.setMeasurementJacobian(ekf::pendulum::measurementJacobian);

    // =========================================================================
    // Create UKF
    // =========================================================================
    ukf::UnscentedKalmanFilter ukf_filter(state_dim, meas_dim);

    ukf_filter.setStateFunction([g, L, b](const Eigen::VectorXd& x, double dt_local) {
        return ukf::pendulum::stateTransition(x, dt_local, g, L, b);
    });

    ukf_filter.setMeasurementFunction(ukf::pendulum::measurementFunction);

    // =========================================================================
    // Set Initial Conditions and Noise Parameters
    // =========================================================================
    Eigen::VectorXd x0(state_dim);
    x0 << 0.5, 0.0;  // Initial guess: θ=0.5 rad, θ̇=0

    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(state_dim, state_dim);
    P0 *= 0.1;  // Initial uncertainty

    Eigen::MatrixXd Q(state_dim, state_dim);
    Q << 1e-4, 0,
         0, 1e-4;

    Eigen::MatrixXd R(meas_dim, meas_dim);
    R << 0.01;  // Measurement noise variance

    // Configure EKF
    ekf_filter.setState(x0);
    ekf_filter.setCovariance(P0);
    ekf_filter.setProcessNoise(Q);
    ekf_filter.setMeasurementNoise(R);

    // Configure UKF
    ukf_filter.setState(x0);
    ukf_filter.setCovariance(P0);
    ukf_filter.setProcessNoise(Q);
    ukf_filter.setMeasurementNoise(R);

    // =========================================================================
    // Generate True Trajectory and Measurements
    // =========================================================================
    std::mt19937 rng(42);
    std::normal_distribution<double> meas_noise(0.0, std::sqrt(R(0, 0)));

    // True initial state (different from filter initial guess)
    Eigen::VectorXd x_true(state_dim);
    x_true << M_PI / 4, 0.0;  // 45 degrees, stationary

    const int num_steps = 500;
    std::vector<double> true_angles(num_steps);
    std::vector<double> measurements(num_steps);

    std::cout << "Generating true trajectory and measurements...\n\n";

    Eigen::VectorXd x_current = x_true;
    for (int k = 0; k < num_steps; ++k) {
        true_angles[k] = x_current(0);
        measurements[k] = x_current(0) + meas_noise(rng);

        // Propagate true state
        x_current = ekf::pendulum::stateTransition(x_current, dt, g, L, b);
    }

    // =========================================================================
    // Run Filters and Compare
    // =========================================================================
    std::cout << "Running EKF and UKF...\n\n";

    std::vector<double> ekf_estimates(num_steps);
    std::vector<double> ukf_estimates(num_steps);
    std::vector<double> ekf_errors(num_steps);
    std::vector<double> ukf_errors(num_steps);

    for (int k = 0; k < num_steps; ++k) {
        // Predict
        ekf_filter.predict(dt);
        ukf_filter.predict(dt);

        // Update with measurement
        Eigen::VectorXd z(meas_dim);
        z << measurements[k];

        ekf_filter.update(z);
        ukf_filter.update(z);

        // Store results
        ekf_estimates[k] = ekf_filter.getState()(0);
        ukf_estimates[k] = ukf_filter.getState()(0);
        ekf_errors[k] = std::abs(ekf_estimates[k] - true_angles[k]);
        ukf_errors[k] = std::abs(ukf_estimates[k] - true_angles[k]);
    }

    // =========================================================================
    // Print Results
    // =========================================================================
    std::cout << "Results (showing every 50th step):\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << std::setw(6) << "Step"
              << std::setw(12) << "True θ"
              << std::setw(12) << "Measured"
              << std::setw(12) << "EKF θ"
              << std::setw(12) << "UKF θ"
              << std::setw(12) << "EKF Err"
              << std::setw(12) << "UKF Err" << "\n";
    std::cout << "-------------------------------------------------------------------\n";

    for (int k = 0; k < num_steps; k += 50) {
        std::cout << std::setw(6) << k
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << true_angles[k]
                  << std::setw(12) << measurements[k]
                  << std::setw(12) << ekf_estimates[k]
                  << std::setw(12) << ukf_estimates[k]
                  << std::setw(12) << ekf_errors[k]
                  << std::setw(12) << ukf_errors[k] << "\n";
    }

    // =========================================================================
    // Compute Statistics
    // =========================================================================
    double ekf_rmse = 0, ukf_rmse = 0;
    double ekf_max_err = 0, ukf_max_err = 0;

    for (int k = 0; k < num_steps; ++k) {
        ekf_rmse += ekf_errors[k] * ekf_errors[k];
        ukf_rmse += ukf_errors[k] * ukf_errors[k];
        ekf_max_err = std::max(ekf_max_err, ekf_errors[k]);
        ukf_max_err = std::max(ukf_max_err, ukf_errors[k]);
    }
    ekf_rmse = std::sqrt(ekf_rmse / num_steps);
    ukf_rmse = std::sqrt(ukf_rmse / num_steps);

    std::cout << "\n=========================================================\n";
    std::cout << "                    PERFORMANCE SUMMARY\n";
    std::cout << "=========================================================\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "EKF RMSE:     " << ekf_rmse << " rad\n";
    std::cout << "UKF RMSE:     " << ukf_rmse << " rad\n";
    std::cout << "EKF Max Err:  " << ekf_max_err << " rad\n";
    std::cout << "UKF Max Err:  " << ukf_max_err << " rad\n\n";

    std::cout << "Final States:\n";
    std::cout << "  EKF: θ = " << ekf_filter.getState()(0)
              << ", θ̇ = " << ekf_filter.getState()(1) << "\n";
    std::cout << "  UKF: θ = " << ukf_filter.getState()(0)
              << ", θ̇ = " << ukf_filter.getState()(1) << "\n";
    std::cout << "  True: θ = " << true_angles[num_steps-1]
              << ", θ̇ = " << x_current(1) << "\n\n";

    std::cout << "=========================================================\n";
    std::cout << "Both EKF and UKF successfully track the nonlinear pendulum!\n";
    std::cout << "The UKF typically provides slightly better accuracy for\n";
    std::cout << "highly nonlinear systems without requiring Jacobians.\n";
    std::cout << "=========================================================\n";

    return 0;
}
