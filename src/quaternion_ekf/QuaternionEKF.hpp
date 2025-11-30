/**
 * @file QuaternionEKF.hpp
 * @brief Quaternion-based Extended Kalman Filter for 3D Orientation Estimation
 *
 * This implementation uses quaternions to represent 3D orientation, avoiding
 * gimbal lock and providing smooth interpolation. It's commonly used for:
 * - IMU sensor fusion (accelerometer + gyroscope + magnetometer)
 * - Spacecraft attitude estimation
 * - Drone/robot orientation tracking
 *
 * Mathematical Foundation:
 * ========================
 * Quaternion: q = [qw, qx, qy, qz]' where qw is scalar part
 * Constraint: ||q|| = 1 (unit quaternion)
 *
 * State Vector: x = [qw, qx, qy, qz, ωx, ωy, ωz]'
 * - q: orientation quaternion
 * - ω: angular velocity (body frame)
 *
 * Dynamics:
 * q̇ = 0.5 * q ⊗ [0, ω]
 * where ⊗ is quaternion multiplication
 *
 * Measurement:
 * Typically gravity vector and/or magnetic field in body frame
 */

#ifndef QUATERNION_EKF_HPP
#define QUATERNION_EKF_HPP

#include <Eigen/Dense>
#include <cmath>

namespace quaternion_ekf {

/**
 * @class QuaternionEKF
 * @brief EKF for 3D orientation estimation using quaternions
 *
 * This filter estimates both orientation (as quaternion) and angular velocity.
 * It handles the quaternion normalization constraint and uses proper
 * quaternion kinematics.
 */
class QuaternionEKF {
public:
    /**
     * @brief Constructor
     * Initializes state to identity quaternion and zero angular velocity
     */
    QuaternionEKF()
    {
        // State: [qw, qx, qy, qz, wx, wy, wz]
        x_ = Eigen::VectorXd::Zero(7);
        x_(0) = 1.0;  // Identity quaternion (no rotation)

        // Initial covariance
        P_ = Eigen::MatrixXd::Identity(7, 7);
        P_.block<4, 4>(0, 0) *= 0.01;  // Small initial quaternion uncertainty
        P_.block<3, 3>(4, 4) *= 0.1;   // Angular velocity uncertainty

        // Default noise parameters
        Q_ = Eigen::MatrixXd::Identity(7, 7);
        Q_.block<4, 4>(0, 0) *= 1e-6;  // Quaternion process noise
        Q_.block<3, 3>(4, 4) *= 1e-3;  // Gyro random walk

        R_ = Eigen::MatrixXd::Identity(3, 3) * 0.1;  // Measurement noise
    }

    // =========================================================================
    // Setters
    // =========================================================================

    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
    void setState(const Eigen::VectorXd& x) { x_ = x; normalizeQuaternion(); }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

    // =========================================================================
    // Getters
    // =========================================================================

    const Eigen::VectorXd& getState() const { return x_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }

    /**
     * @brief Get quaternion part of state
     */
    Eigen::Vector4d getQuaternion() const { return x_.head<4>(); }

    /**
     * @brief Get angular velocity part of state
     */
    Eigen::Vector3d getAngularVelocity() const { return x_.tail<3>(); }

    /**
     * @brief Convert quaternion to Euler angles (roll, pitch, yaw)
     * @return Euler angles in radians [roll, pitch, yaw]
     */
    Eigen::Vector3d getEulerAngles() const
    {
        double qw = x_(0), qx = x_(1), qy = x_(2), qz = x_(3);

        // Roll (x-axis rotation)
        double sinr_cosp = 2.0 * (qw * qx + qy * qz);
        double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
        double roll = std::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        double sinp = 2.0 * (qw * qy - qz * qx);
        double pitch;
        if (std::abs(sinp) >= 1)
            pitch = std::copysign(M_PI / 2, sinp);  // Use 90 degrees if out of range
        else
            pitch = std::asin(sinp);

        // Yaw (z-axis rotation)
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        return Eigen::Vector3d(roll, pitch, yaw);
    }

    // =========================================================================
    // Core Filter Operations
    // =========================================================================

    /**
     * @brief Prediction Step with gyroscope measurement
     *
     * Quaternion Kinematics:
     * q̇ = 0.5 * Ω(ω) * q
     *
     * where Ω(ω) is the skew-symmetric matrix:
     * Ω = [0   -ωx  -ωy  -ωz]
     *     [ωx   0    ωz  -ωy]
     *     [ωy  -ωz   0    ωx]
     *     [ωz   ωy  -ωx   0 ]
     *
     * @param gyro Gyroscope measurement [wx, wy, wz] in rad/s
     * @param dt Time step in seconds
     */
    void predict(const Eigen::Vector3d& gyro, double dt)
    {
        // Update angular velocity estimate with gyro measurement
        x_.tail<3>() = gyro;

        // Get current quaternion and angular velocity
        Eigen::Vector4d q = x_.head<4>();
        Eigen::Vector3d w = x_.tail<3>();

        // Quaternion derivative: q̇ = 0.5 * q ⊗ [0, ω]
        Eigen::Vector4d q_dot = 0.5 * quaternionMultiply(q, Eigen::Vector4d(0, w(0), w(1), w(2)));

        // Integrate quaternion
        x_.head<4>() = q + q_dot * dt;

        // Normalize quaternion to maintain unit constraint
        normalizeQuaternion();

        // Compute Jacobian for covariance propagation
        Eigen::MatrixXd F = computeStateJacobian(dt);

        // Propagate covariance
        P_ = F * P_ * F.transpose() + Q_;
    }

    /**
     * @brief Update Step with accelerometer measurement (gravity reference)
     *
     * The accelerometer measures gravity in the body frame when the device
     * is stationary or moving at constant velocity. This provides a
     * reference for roll and pitch.
     *
     * Expected gravity in body frame: g_body = q* ⊗ [0,0,0,g] ⊗ q
     *
     * @param accel Accelerometer measurement (normalized)
     */
    void updateWithAccelerometer(const Eigen::Vector3d& accel)
    {
        // Normalize accelerometer reading
        Eigen::Vector3d accel_norm = accel.normalized();

        // Predicted gravity direction in body frame
        Eigen::Vector3d g_pred = rotateVectorByQuaternion(Eigen::Vector3d(0, 0, 1), x_.head<4>().normalized());

        // Innovation
        Eigen::Vector3d y = accel_norm - g_pred;

        // Compute measurement Jacobian
        Eigen::MatrixXd H = computeAccelMeasurementJacobian();

        // Standard Kalman update
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        normalizeQuaternion();

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
        P_ = (I - K * H) * P_;
    }

    /**
     * @brief Update Step with magnetometer measurement (heading reference)
     *
     * The magnetometer provides a reference for yaw/heading by measuring
     * Earth's magnetic field.
     *
     * @param mag Magnetometer measurement (normalized)
     * @param mag_ref Reference magnetic field direction in world frame
     */
    void updateWithMagnetometer(const Eigen::Vector3d& mag,
                                const Eigen::Vector3d& mag_ref = Eigen::Vector3d(1, 0, 0))
    {
        // Normalize magnetometer reading
        Eigen::Vector3d mag_norm = mag.normalized();

        // Predicted magnetic field direction in body frame
        Eigen::Vector3d mag_pred = rotateVectorByQuaternion(mag_ref, x_.head<4>().normalized());

        // Only use horizontal component for yaw correction
        // Project both to horizontal plane
        Eigen::Vector3d mag_norm_horiz = mag_norm;
        mag_norm_horiz(2) = 0;
        mag_norm_horiz.normalize();

        Eigen::Vector3d mag_pred_horiz = mag_pred;
        mag_pred_horiz(2) = 0;
        if (mag_pred_horiz.norm() > 0.01) {
            mag_pred_horiz.normalize();
        }

        // Innovation (simplified - just use full 3D for now)
        Eigen::Vector3d y = mag_norm - mag_pred;

        // Use same measurement model as accelerometer
        Eigen::MatrixXd H = computeAccelMeasurementJacobian();

        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        normalizeQuaternion();

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
        P_ = (I - K * H) * P_;
    }

private:
    Eigen::VectorXd x_;  ///< State: [qw, qx, qy, qz, wx, wy, wz]
    Eigen::MatrixXd P_;  ///< State covariance (7x7)
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd R_;  ///< Measurement noise covariance

    /**
     * @brief Normalize quaternion part of state
     */
    void normalizeQuaternion()
    {
        double norm = x_.head<4>().norm();
        if (norm > 1e-10) {
            x_.head<4>() /= norm;
        }
        // Ensure positive scalar part (hemisphere constraint)
        if (x_(0) < 0) {
            x_.head<4>() *= -1;
        }
    }

    /**
     * @brief Quaternion multiplication: q1 ⊗ q2
     * Convention: [w, x, y, z]
     */
    static Eigen::Vector4d quaternionMultiply(const Eigen::Vector4d& q1,
                                              const Eigen::Vector4d& q2)
    {
        Eigen::Vector4d result;
        result(0) = q1(0)*q2(0) - q1(1)*q2(1) - q1(2)*q2(2) - q1(3)*q2(3);
        result(1) = q1(0)*q2(1) + q1(1)*q2(0) + q1(2)*q2(3) - q1(3)*q2(2);
        result(2) = q1(0)*q2(2) - q1(1)*q2(3) + q1(2)*q2(0) + q1(3)*q2(1);
        result(3) = q1(0)*q2(3) + q1(1)*q2(2) - q1(2)*q2(1) + q1(3)*q2(0);
        return result;
    }

    /**
     * @brief Rotate a vector by a quaternion: v' = q ⊗ [0,v] ⊗ q*
     */
    static Eigen::Vector3d rotateVectorByQuaternion(const Eigen::Vector3d& v,
                                                    const Eigen::Vector4d& q)
    {
        // Quaternion conjugate
        Eigen::Vector4d q_conj(q(0), -q(1), -q(2), -q(3));

        // v as quaternion [0, vx, vy, vz]
        Eigen::Vector4d v_quat(0, v(0), v(1), v(2));

        // q ⊗ v ⊗ q*
        Eigen::Vector4d result = quaternionMultiply(quaternionMultiply(q, v_quat), q_conj);

        return Eigen::Vector3d(result(1), result(2), result(3));
    }

    /**
     * @brief Compute state transition Jacobian
     */
    Eigen::MatrixXd computeStateJacobian(double dt)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(7, 7);

        double wx = x_(4), wy = x_(5), wz = x_(6);

        // Jacobian of quaternion dynamics w.r.t quaternion
        // ∂(q + 0.5*Ω*q*dt)/∂q = I + 0.5*Ω*dt
        Eigen::Matrix4d Omega;
        Omega << 0, -wx, -wy, -wz,
                 wx, 0, wz, -wy,
                 wy, -wz, 0, wx,
                 wz, wy, -wx, 0;

        F.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity() + 0.5 * Omega * dt;

        // Jacobian of quaternion dynamics w.r.t angular velocity
        double qw = x_(0), qx = x_(1), qy = x_(2), qz = x_(3);
        Eigen::Matrix<double, 4, 3> G;
        G << -qx, -qy, -qz,
              qw, -qz,  qy,
              qz,  qw, -qx,
             -qy,  qx,  qw;
        F.block<4, 3>(0, 4) = 0.5 * G * dt;

        return F;
    }

    /**
     * @brief Compute measurement Jacobian for accelerometer
     */
    Eigen::MatrixXd computeAccelMeasurementJacobian()
    {
        // Numerical Jacobian (simplified)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 7);

        double eps = 1e-6;
        Eigen::Vector4d q = x_.head<4>().normalized();
        Eigen::Vector3d g_world(0, 0, 1);
        Eigen::Vector3d h0 = rotateVectorByQuaternion(g_world, q);

        for (int i = 0; i < 4; ++i) {
            Eigen::Vector4d q_pert = q;
            q_pert(i) += eps;
            q_pert.normalize();
            Eigen::Vector3d h_pert = rotateVectorByQuaternion(g_world, q_pert);
            H.col(i) = (h_pert - h0) / eps;
        }

        return H;
    }
};

} // namespace quaternion_ekf

#endif // QUATERNION_EKF_HPP
