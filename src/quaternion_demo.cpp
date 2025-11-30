/**
 * @file quaternion_demo.cpp
 * @brief Demonstrates Quaternion EKF and UKF for 3D orientation estimation
 *
 * This demo simulates an IMU-like sensor fusion problem where we estimate
 * 3D orientation from gyroscope, accelerometer, and magnetometer readings.
 *
 * Application: Attitude/Heading Reference System (AHRS)
 * - Gyroscope: Provides angular velocity (integrates to orientation)
 * - Accelerometer: Provides gravity reference (roll/pitch correction)
 * - Magnetometer: Provides magnetic north reference (yaw correction)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>

#include "quaternion_ekf/QuaternionEKF.hpp"
#include "quaternion_ukf/QuaternionUKF.hpp"

// Define PI for portability (not guaranteed in all compilers)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Convert degrees to radians
 */
inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

/**
 * @brief Convert radians to degrees
 */
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

/**
 * @brief Create quaternion from Euler angles (ZYX convention)
 */
Eigen::Vector4d eulerToQuaternion(double roll, double pitch, double yaw)
{
    double cr = std::cos(roll / 2), sr = std::sin(roll / 2);
    double cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
    double cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);

    Eigen::Vector4d q;
    q(0) = cr * cp * cy + sr * sp * sy;  // w
    q(1) = sr * cp * cy - cr * sp * sy;  // x
    q(2) = cr * sp * cy + sr * cp * sy;  // y
    q(3) = cr * cp * sy - sr * sp * cy;  // z
    return q.normalized();
}

/**
 * @brief Rotate a vector by quaternion
 */
Eigen::Vector3d rotateByQuaternion(const Eigen::Vector3d& v, const Eigen::Vector4d& q)
{
    Eigen::Vector4d q_conj(q(0), -q(1), -q(2), -q(3));
    Eigen::Vector4d v_quat(0, v(0), v(1), v(2));

    // q ⊗ v ⊗ q*
    auto qmul = [](const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
        Eigen::Vector4d result;
        result(0) = a(0)*b(0) - a(1)*b(1) - a(2)*b(2) - a(3)*b(3);
        result(1) = a(0)*b(1) + a(1)*b(0) + a(2)*b(3) - a(3)*b(2);
        result(2) = a(0)*b(2) - a(1)*b(3) + a(2)*b(0) + a(3)*b(1);
        result(3) = a(0)*b(3) + a(1)*b(2) - a(2)*b(1) + a(3)*b(0);
        return result;
    };

    Eigen::Vector4d result = qmul(qmul(q, v_quat), q_conj);
    return Eigen::Vector3d(result(1), result(2), result(3));
}

int main()
{
    std::cout << "=========================================================\n";
    std::cout << "   Quaternion EKF vs UKF: 3D Orientation Estimation\n";
    std::cout << "=========================================================\n\n";

    // =========================================================================
    // Simulation Parameters
    // =========================================================================
    const double dt = 0.01;  // 100 Hz IMU rate
    const int num_steps = 1000;  // 10 seconds of data

    // Noise parameters
    const double gyro_noise_std = deg2rad(0.5);     // Gyro noise (deg/s)
    const double accel_noise_std = 0.05;            // Accelerometer noise (g)
    const double mag_noise_std = 0.05;              // Magnetometer noise

    std::mt19937 rng(42);
    std::normal_distribution<double> gyro_noise(0, gyro_noise_std);
    std::normal_distribution<double> accel_noise(0, accel_noise_std);
    std::normal_distribution<double> mag_noise(0, mag_noise_std);

    // =========================================================================
    // Create Filters
    // =========================================================================
    quaternion_ekf::QuaternionEKF ekf;
    quaternion_ukf::QuaternionUKF ukf;

    // Set noise parameters
    Eigen::MatrixXd Q_ekf = Eigen::MatrixXd::Identity(7, 7);
    Q_ekf.block<4, 4>(0, 0) *= 1e-6;
    Q_ekf.block<3, 3>(4, 4) *= 1e-4;
    ekf.setProcessNoise(Q_ekf);

    Eigen::MatrixXd R_ekf = Eigen::MatrixXd::Identity(3, 3) * 0.01;
    ekf.setMeasurementNoise(R_ekf);

    Eigen::MatrixXd Q_ukf = Eigen::MatrixXd::Identity(6, 6);
    Q_ukf.block<3, 3>(0, 0) *= 1e-5;
    Q_ukf.block<3, 3>(3, 3) *= 1e-4;
    ukf.setProcessNoise(Q_ukf);

    Eigen::MatrixXd R_ukf = Eigen::MatrixXd::Identity(3, 3) * 0.01;
    ukf.setMeasurementNoise(R_ukf);

    // =========================================================================
    // Define True Motion Profile
    // =========================================================================
    // Simulate a rotating platform:
    // - Constant yaw rotation (turning)
    // - Sinusoidal roll motion
    // - Small pitch oscillation

    auto getTrueAngularVelocity = [](double t) -> Eigen::Vector3d {
        return Eigen::Vector3d(
            0.3 * std::cos(0.5 * t),        // Roll rate
            0.1 * std::sin(0.3 * t),        // Pitch rate
            0.2                              // Constant yaw rate
        );
    };

    // =========================================================================
    // Run Simulation
    // =========================================================================
    std::cout << "Running 10-second IMU simulation at 100 Hz...\n\n";

    // True state
    Eigen::Vector4d q_true = eulerToQuaternion(0, 0, 0);  // Start at identity
    Eigen::Vector3d w_true = Eigen::Vector3d::Zero();

    // Storage for results
    std::vector<Eigen::Vector3d> true_euler(num_steps);
    std::vector<Eigen::Vector3d> ekf_euler(num_steps);
    std::vector<Eigen::Vector3d> ukf_euler(num_steps);

    // Reference vectors (world frame)
    Eigen::Vector3d gravity_world(0, 0, 1);   // Gravity points up in world
    Eigen::Vector3d mag_world(1, 0, 0);       // Magnetic north

    for (int k = 0; k < num_steps; ++k) {
        double t = k * dt;

        // Get true angular velocity
        w_true = getTrueAngularVelocity(t);

        // Integrate true quaternion
        Eigen::Vector4d q_dot = 0.5 * Eigen::Vector4d(
            -q_true(1)*w_true(0) - q_true(2)*w_true(1) - q_true(3)*w_true(2),
             q_true(0)*w_true(0) + q_true(2)*w_true(2) - q_true(3)*w_true(1),
             q_true(0)*w_true(1) - q_true(1)*w_true(2) + q_true(3)*w_true(0),
             q_true(0)*w_true(2) + q_true(1)*w_true(1) - q_true(2)*w_true(0)
        );
        q_true = (q_true + q_dot * dt).normalized();

        // Generate sensor measurements
        // Gyro (angular velocity + noise)
        Eigen::Vector3d gyro_meas = w_true;
        gyro_meas(0) += gyro_noise(rng);
        gyro_meas(1) += gyro_noise(rng);
        gyro_meas(2) += gyro_noise(rng);

        // Accelerometer (gravity in body frame + noise)
        Eigen::Vector3d accel_meas = rotateByQuaternion(gravity_world, q_true);
        accel_meas(0) += accel_noise(rng);
        accel_meas(1) += accel_noise(rng);
        accel_meas(2) += accel_noise(rng);

        // Magnetometer (magnetic field in body frame + noise)
        Eigen::Vector3d mag_meas = rotateByQuaternion(mag_world, q_true);
        mag_meas(0) += mag_noise(rng);
        mag_meas(1) += mag_noise(rng);
        mag_meas(2) += mag_noise(rng);

        // =====================================================================
        // EKF Update
        // =====================================================================
        ekf.predict(gyro_meas, dt);
        ekf.updateWithAccelerometer(accel_meas);
        if (k % 10 == 0) {  // Magnetometer at 10 Hz
            ekf.updateWithMagnetometer(mag_meas);
        }

        // =====================================================================
        // UKF Update
        // =====================================================================
        ukf.predict(gyro_meas, dt);
        ukf.updateWithAccelerometer(accel_meas);
        if (k % 10 == 0) {  // Magnetometer at 10 Hz
            ukf.updateWithMagnetometer(mag_meas);
        }

        // Store Euler angles for comparison
        // True euler from quaternion
        double qw = q_true(0), qx = q_true(1), qy = q_true(2), qz = q_true(3);
        true_euler[k] = Eigen::Vector3d(
            std::atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)),
            std::asin(std::clamp(2*(qw*qy - qz*qx), -1.0, 1.0)),
            std::atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        );

        ekf_euler[k] = ekf.getEulerAngles();
        ukf_euler[k] = ukf.getEulerAngles();
    }

    // =========================================================================
    // Print Results
    // =========================================================================
    std::cout << "Results (showing every 100th step):\n";
    std::cout << "-----------------------------------------------------------------------------\n";
    std::cout << std::setw(6) << "Step"
              << " | True (R,P,Y) deg | EKF (R,P,Y) deg | UKF (R,P,Y) deg |\n";
    std::cout << "-----------------------------------------------------------------------------\n";

    for (int k = 0; k < num_steps; k += 100) {
        std::cout << std::setw(6) << k << " | ";
        std::cout << std::fixed << std::setprecision(1);

        // True
        std::cout << std::setw(5) << rad2deg(true_euler[k](0)) << ","
                  << std::setw(5) << rad2deg(true_euler[k](1)) << ","
                  << std::setw(5) << rad2deg(true_euler[k](2)) << " | ";

        // EKF
        std::cout << std::setw(5) << rad2deg(ekf_euler[k](0)) << ","
                  << std::setw(5) << rad2deg(ekf_euler[k](1)) << ","
                  << std::setw(5) << rad2deg(ekf_euler[k](2)) << " | ";

        // UKF
        std::cout << std::setw(5) << rad2deg(ukf_euler[k](0)) << ","
                  << std::setw(5) << rad2deg(ukf_euler[k](1)) << ","
                  << std::setw(5) << rad2deg(ukf_euler[k](2)) << " |\n";
    }

    // =========================================================================
    // Compute Statistics
    // =========================================================================
    Eigen::Vector3d ekf_rmse = Eigen::Vector3d::Zero();
    Eigen::Vector3d ukf_rmse = Eigen::Vector3d::Zero();

    for (int k = 0; k < num_steps; ++k) {
        Eigen::Vector3d ekf_err = ekf_euler[k] - true_euler[k];
        Eigen::Vector3d ukf_err = ukf_euler[k] - true_euler[k];

        // Handle angle wrapping for yaw
        while (ekf_err(2) > M_PI) ekf_err(2) -= 2*M_PI;
        while (ekf_err(2) < -M_PI) ekf_err(2) += 2*M_PI;
        while (ukf_err(2) > M_PI) ukf_err(2) -= 2*M_PI;
        while (ukf_err(2) < -M_PI) ukf_err(2) += 2*M_PI;

        ekf_rmse += ekf_err.cwiseProduct(ekf_err);
        ukf_rmse += ukf_err.cwiseProduct(ukf_err);
    }
    ekf_rmse = (ekf_rmse / num_steps).cwiseSqrt();
    ukf_rmse = (ukf_rmse / num_steps).cwiseSqrt();

    std::cout << "\n=========================================================\n";
    std::cout << "                 PERFORMANCE SUMMARY\n";
    std::cout << "=========================================================\n\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "EKF RMSE (Roll, Pitch, Yaw): "
              << rad2deg(ekf_rmse(0)) << "°, "
              << rad2deg(ekf_rmse(1)) << "°, "
              << rad2deg(ekf_rmse(2)) << "°\n";

    std::cout << "UKF RMSE (Roll, Pitch, Yaw): "
              << rad2deg(ukf_rmse(0)) << "°, "
              << rad2deg(ukf_rmse(1)) << "°, "
              << rad2deg(ukf_rmse(2)) << "°\n\n";

    std::cout << "Final Quaternions:\n";
    auto q_ekf = ekf.getQuaternion();
    auto q_ukf = ukf.getQuaternion();
    std::cout << "  True: [" << q_true.transpose() << "]\n";
    std::cout << "  EKF:  [" << q_ekf.transpose() << "]\n";
    std::cout << "  UKF:  [" << q_ukf.transpose() << "]\n\n";

    std::cout << "=========================================================\n";
    std::cout << "Both Quaternion EKF and UKF successfully track 3D\n";
    std::cout << "orientation by fusing gyroscope, accelerometer, and\n";
    std::cout << "magnetometer data - commonly used in AHRS/IMU systems!\n";
    std::cout << "=========================================================\n";

    return 0;
}
