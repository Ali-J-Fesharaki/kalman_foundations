/**
 * @file QuaternionUKF.hpp
 * @brief Quaternion-based Unscented Kalman Filter for 3D Orientation Estimation
 *
 * This implementation combines the UKF's superior handling of nonlinear systems
 * with quaternion representation for 3D orientation. The UKF avoids computing
 * Jacobians and provides better accuracy for the highly nonlinear quaternion
 * dynamics.
 *
 * Key Challenge: Quaternion Sigma Points
 * ======================================
 * Standard UKF assumes additive Gaussian noise in Euclidean space, but
 * quaternions live on the unit sphere S³. Special care is needed:
 *
 * 1. Sigma points must be generated on the quaternion manifold
 * 2. Mean computation uses quaternion averaging
 * 3. "Errors" are represented as small rotation vectors
 *
 * State Vector: x = [qw, qx, qy, qz, ωx, ωy, ωz]'
 * - q: orientation quaternion (unit constraint)
 * - ω: angular velocity (body frame)
 *
 * Applications:
 * - IMU/AHRS systems
 * - Spacecraft attitude determination
 * - Drone flight controllers
 * - VR/AR head tracking
 */

#ifndef QUATERNION_UKF_HPP
#define QUATERNION_UKF_HPP

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace quaternion_ukf {

/**
 * @class QuaternionUKF
 * @brief UKF for 3D orientation estimation using quaternions
 *
 * This filter properly handles the quaternion unit constraint by
 * working in the tangent space for sigma point generation and
 * projecting back onto the unit sphere.
 */
class QuaternionUKF {
public:
    /**
     * @brief Constructor
     * @param alpha Spread of sigma points (default 1e-2)
     * @param beta Prior knowledge (2 for Gaussian)
     * @param kappa Secondary scaling (0 is typical)
     */
    QuaternionUKF(double alpha = 1e-2, double beta = 2.0, double kappa = 0.0)
        : alpha_(alpha), beta_(beta), kappa_(kappa)
    {
        // State dimension: 7 (4 quaternion + 3 angular velocity)
        // But for sigma points we use 6 (3 rotation error + 3 angular velocity)
        state_dim_ = 6;  // Error state dimension
        meas_dim_ = 3;

        // Initialize state
        x_ = Eigen::VectorXd::Zero(7);
        x_(0) = 1.0;  // Identity quaternion

        // Covariance for error state (6x6)
        P_ = Eigen::MatrixXd::Identity(6, 6);
        P_.block<3, 3>(0, 0) *= 0.01;  // Rotation error covariance
        P_.block<3, 3>(3, 3) *= 0.1;   // Angular velocity covariance

        // Process noise
        Q_ = Eigen::MatrixXd::Identity(6, 6);
        Q_.block<3, 3>(0, 0) *= 1e-5;  // Rotation process noise
        Q_.block<3, 3>(3, 3) *= 1e-3;  // Gyro random walk

        // Measurement noise
        R_ = Eigen::MatrixXd::Identity(3, 3) * 0.1;

        computeWeights();
    }

    // =========================================================================
    // Setters
    // =========================================================================

    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

    void setState(const Eigen::VectorXd& x)
    {
        x_ = x;
        normalizeQuaternion();
    }

    // =========================================================================
    // Getters
    // =========================================================================

    const Eigen::VectorXd& getState() const { return x_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }
    Eigen::Vector4d getQuaternion() const { return x_.head<4>(); }
    Eigen::Vector3d getAngularVelocity() const { return x_.tail<3>(); }

    /**
     * @brief Convert quaternion to Euler angles
     */
    Eigen::Vector3d getEulerAngles() const
    {
        double qw = x_(0), qx = x_(1), qy = x_(2), qz = x_(3);

        double roll = std::atan2(2.0 * (qw * qx + qy * qz),
                                 1.0 - 2.0 * (qx * qx + qy * qy));

        double sinp = 2.0 * (qw * qy - qz * qx);
        double pitch = std::abs(sinp) >= 1 ?
                       std::copysign(M_PI / 2, sinp) : std::asin(sinp);

        double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy * qy + qz * qz));

        return Eigen::Vector3d(roll, pitch, yaw);
    }

    // =========================================================================
    // Core Filter Operations
    // =========================================================================

    /**
     * @brief Prediction Step using Unscented Transform
     *
     * For quaternions, sigma points are generated in the error state space
     * (rotation vector + angular velocity), then mapped to full state space.
     *
     * @param gyro Gyroscope measurement [wx, wy, wz] in rad/s
     * @param dt Time step in seconds
     */
    void predict(const Eigen::Vector3d& gyro, double dt)
    {
        // Generate sigma points in error state space
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints();

        // Store propagated sigma points (full 7-state)
        std::vector<Eigen::VectorXd> propagated_states(2 * state_dim_ + 1);

        // Propagate each sigma point through dynamics
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            propagated_states[i] = propagateDynamics(sigma_points[i], gyro, dt);
        }

        // Compute mean quaternion using weighted average
        Eigen::Vector4d q_mean = computeQuaternionMean(propagated_states);

        // Compute mean angular velocity (standard weighted average)
        Eigen::Vector3d w_mean = Eigen::Vector3d::Zero();
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            w_mean += Wm_[i] * propagated_states[i].tail<3>();
        }

        // Update state
        x_.head<4>() = q_mean;
        x_.tail<3>() = w_mean;

        // Compute covariance in error state space
        P_ = Eigen::MatrixXd::Zero(6, 6);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd error = computeErrorState(propagated_states[i], x_);
            P_ += Wc_[i] * error * error.transpose();
        }
        P_ += Q_;
    }

    /**
     * @brief Update with accelerometer measurement (gravity reference)
     *
     * @param accel Accelerometer measurement (normalized)
     */
    void updateWithAccelerometer(const Eigen::Vector3d& accel)
    {
        Eigen::Vector3d accel_norm = accel.normalized();

        // Generate sigma points
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints();

        // Transform through measurement function
        std::vector<Eigen::Vector3d> meas_points(2 * state_dim_ + 1);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd full_state = errorStateToFullState(sigma_points[i]);
            Eigen::Vector4d q = full_state.head<4>();
            // Predicted gravity in body frame
            meas_points[i] = rotateVectorByQuaternion(Eigen::Vector3d(0, 0, 1), q);
        }

        // Compute predicted measurement mean
        Eigen::Vector3d z_pred = Eigen::Vector3d::Zero();
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            z_pred += Wm_[i] * meas_points[i];
        }

        // Compute measurement covariance Pzz
        Eigen::Matrix3d Pzz = Eigen::Matrix3d::Zero();
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::Vector3d dz = meas_points[i] - z_pred;
            Pzz += Wc_[i] * dz * dz.transpose();
        }
        Pzz += R_;

        // Compute cross-covariance Pxz
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(6, 3);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd dx = sigma_points[i];  // Already centered
            Eigen::Vector3d dz = meas_points[i] - z_pred;
            Pxz += Wc_[i] * dx * dz.transpose();
        }

        // Kalman gain
        Eigen::MatrixXd K = Pxz * Pzz.inverse();

        // Innovation
        Eigen::Vector3d y = accel_norm - z_pred;

        // Update error state
        Eigen::VectorXd dx = K * y;

        // Apply correction to quaternion via error rotation
        Eigen::Vector3d rot_error = dx.head<3>();
        Eigen::Vector4d dq = rotationVectorToQuaternion(rot_error);
        x_.head<4>() = quaternionMultiply(dq, x_.head<4>());
        normalizeQuaternion();

        // Update angular velocity
        x_.tail<3>() += dx.tail<3>();

        // Update covariance
        P_ = P_ - K * Pzz * K.transpose();
    }

    /**
     * @brief Update with magnetometer measurement
     */
    void updateWithMagnetometer(const Eigen::Vector3d& mag,
                                const Eigen::Vector3d& mag_ref = Eigen::Vector3d(1, 0, 0))
    {
        Eigen::Vector3d mag_norm = mag.normalized();

        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints();

        std::vector<Eigen::Vector3d> meas_points(2 * state_dim_ + 1);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd full_state = errorStateToFullState(sigma_points[i]);
            Eigen::Vector4d q = full_state.head<4>();
            meas_points[i] = rotateVectorByQuaternion(mag_ref, q);
        }

        Eigen::Vector3d z_pred = Eigen::Vector3d::Zero();
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            z_pred += Wm_[i] * meas_points[i];
        }

        Eigen::Matrix3d Pzz = Eigen::Matrix3d::Zero();
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::Vector3d dz = meas_points[i] - z_pred;
            Pzz += Wc_[i] * dz * dz.transpose();
        }
        Pzz += R_;

        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(6, 3);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd dx = sigma_points[i];
            Eigen::Vector3d dz = meas_points[i] - z_pred;
            Pxz += Wc_[i] * dx * dz.transpose();
        }

        Eigen::MatrixXd K = Pxz * Pzz.inverse();
        Eigen::Vector3d y = mag_norm - z_pred;
        Eigen::VectorXd dx = K * y;

        Eigen::Vector3d rot_error = dx.head<3>();
        Eigen::Vector4d dq = rotationVectorToQuaternion(rot_error);
        x_.head<4>() = quaternionMultiply(dq, x_.head<4>());
        normalizeQuaternion();

        x_.tail<3>() += dx.tail<3>();
        P_ = P_ - K * Pzz * K.transpose();
    }

private:
    int state_dim_;   ///< Error state dimension (6)
    int meas_dim_;    ///< Measurement dimension (3)

    double alpha_, beta_, kappa_, lambda_;
    std::vector<double> Wm_, Wc_;

    Eigen::VectorXd x_;  ///< Full state [qw,qx,qy,qz,wx,wy,wz]
    Eigen::MatrixXd P_;  ///< Error state covariance (6x6)
    Eigen::MatrixXd Q_;  ///< Process noise (6x6)
    Eigen::MatrixXd R_;  ///< Measurement noise (3x3)

    void computeWeights()
    {
        lambda_ = alpha_ * alpha_ * (state_dim_ + kappa_) - state_dim_;

        Wm_.resize(2 * state_dim_ + 1);
        Wc_.resize(2 * state_dim_ + 1);

        Wm_[0] = lambda_ / (state_dim_ + lambda_);
        Wc_[0] = Wm_[0] + (1 - alpha_ * alpha_ + beta_);

        for (int i = 1; i <= 2 * state_dim_; ++i) {
            Wm_[i] = 1.0 / (2.0 * (state_dim_ + lambda_));
            Wc_[i] = Wm_[i];
        }
    }

    void normalizeQuaternion()
    {
        double norm = x_.head<4>().norm();
        if (norm > 1e-10) x_.head<4>() /= norm;
        if (x_(0) < 0) x_.head<4>() *= -1;
    }

    /**
     * @brief Generate sigma points in error state space
     */
    std::vector<Eigen::VectorXd> generateSigmaPoints()
    {
        std::vector<Eigen::VectorXd> sigma_points(2 * state_dim_ + 1);

        double scale = std::sqrt(state_dim_ + lambda_);
        Eigen::LLT<Eigen::MatrixXd> llt(P_);
        Eigen::MatrixXd L = llt.matrixL();
        L *= scale;

        // Center point (zero error)
        sigma_points[0] = Eigen::VectorXd::Zero(6);

        for (int i = 0; i < state_dim_; ++i) {
            sigma_points[i + 1] = L.col(i);
            sigma_points[i + 1 + state_dim_] = -L.col(i);
        }

        return sigma_points;
    }

    /**
     * @brief Convert error state to full state
     */
    Eigen::VectorXd errorStateToFullState(const Eigen::VectorXd& error)
    {
        Eigen::VectorXd full_state(7);

        // Apply rotation error to quaternion
        Eigen::Vector3d rot_error = error.head<3>();
        Eigen::Vector4d dq = rotationVectorToQuaternion(rot_error);
        full_state.head<4>() = quaternionMultiply(dq, x_.head<4>());

        // Add angular velocity error
        full_state.tail<3>() = x_.tail<3>() + error.tail<3>();

        return full_state;
    }

    /**
     * @brief Compute error state between two full states
     */
    Eigen::VectorXd computeErrorState(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& mean)
    {
        Eigen::VectorXd error(6);

        // Quaternion error as rotation vector
        Eigen::Vector4d q = state.head<4>();
        Eigen::Vector4d q_mean = mean.head<4>();
        Eigen::Vector4d q_error = quaternionMultiply(q, quaternionConjugate(q_mean));
        error.head<3>() = quaternionToRotationVector(q_error);

        // Angular velocity error
        error.tail<3>() = state.tail<3>() - mean.tail<3>();

        return error;
    }

    /**
     * @brief Propagate dynamics for one sigma point
     */
    Eigen::VectorXd propagateDynamics(const Eigen::VectorXd& error,
                                      const Eigen::Vector3d& gyro, double dt)
    {
        Eigen::VectorXd full_state = errorStateToFullState(error);

        // Update angular velocity with gyro
        full_state.tail<3>() = gyro;

        // Quaternion kinematics
        Eigen::Vector4d q = full_state.head<4>();
        Eigen::Vector3d w = full_state.tail<3>();
        Eigen::Vector4d q_dot = 0.5 * quaternionMultiply(q, Eigen::Vector4d(0, w(0), w(1), w(2)));

        full_state.head<4>() = q + q_dot * dt;
        double norm = full_state.head<4>().norm();
        if (norm > 1e-10) full_state.head<4>() /= norm;

        return full_state;
    }

    /**
     * @brief Compute mean quaternion from weighted samples
     */
    Eigen::Vector4d computeQuaternionMean(const std::vector<Eigen::VectorXd>& states)
    {
        // Initialize with first quaternion
        Eigen::Vector4d q_mean = states[0].head<4>();

        // Iterative averaging
        for (int iter = 0; iter < 10; ++iter) {
            Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

            for (size_t i = 0; i < states.size(); ++i) {
                Eigen::Vector4d q = states[i].head<4>();
                Eigen::Vector4d q_error = quaternionMultiply(q, quaternionConjugate(q_mean));
                error_sum += Wm_[i] * quaternionToRotationVector(q_error);
            }

            if (error_sum.norm() < 1e-10) break;

            Eigen::Vector4d dq = rotationVectorToQuaternion(error_sum);
            q_mean = quaternionMultiply(dq, q_mean);
            q_mean.normalize();
        }

        return q_mean;
    }

    // Quaternion utilities
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

    static Eigen::Vector4d quaternionConjugate(const Eigen::Vector4d& q)
    {
        return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
    }

    static Eigen::Vector3d rotateVectorByQuaternion(const Eigen::Vector3d& v,
                                                    const Eigen::Vector4d& q)
    {
        Eigen::Vector4d q_conj = quaternionConjugate(q);
        Eigen::Vector4d v_quat(0, v(0), v(1), v(2));
        Eigen::Vector4d result = quaternionMultiply(quaternionMultiply(q, v_quat), q_conj);
        return Eigen::Vector3d(result(1), result(2), result(3));
    }

    static Eigen::Vector4d rotationVectorToQuaternion(const Eigen::Vector3d& rv)
    {
        double angle = rv.norm();
        if (angle < 1e-10) {
            return Eigen::Vector4d(1, 0, 0, 0);
        }
        Eigen::Vector3d axis = rv / angle;
        double s = std::sin(angle / 2);
        double c = std::cos(angle / 2);
        return Eigen::Vector4d(c, s * axis(0), s * axis(1), s * axis(2));
    }

    static Eigen::Vector3d quaternionToRotationVector(const Eigen::Vector4d& q)
    {
        Eigen::Vector4d q_norm = q.normalized();
        if (q_norm(0) < 0) q_norm *= -1;

        double w = q_norm(0);
        Eigen::Vector3d v(q_norm(1), q_norm(2), q_norm(3));
        double sin_half = v.norm();

        if (sin_half < 1e-10) {
            return Eigen::Vector3d::Zero();
        }

        double angle = 2.0 * std::atan2(sin_half, w);
        return (angle / sin_half) * v;
    }
};

} // namespace quaternion_ukf

#endif // QUATERNION_UKF_HPP
