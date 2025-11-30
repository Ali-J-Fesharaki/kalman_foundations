/**
 * @file ExtendedKalmanFilter.hpp
 * @brief Extended Kalman Filter (EKF) Implementation
 *
 * The Extended Kalman Filter handles nonlinear systems by linearizing the
 * system dynamics and measurement functions around the current estimate.
 *
 * Mathematical Foundation:
 * ========================
 * For nonlinear systems:
 *   x[k+1] = f(x[k], u[k]) + w[k]    (nonlinear state transition)
 *   z[k] = h(x[k]) + v[k]             (nonlinear measurement)
 *
 * The EKF linearizes these functions using Jacobians:
 *   F = ∂f/∂x |_{x=x̂}  (State transition Jacobian)
 *   H = ∂h/∂x |_{x=x̂}  (Measurement Jacobian)
 *
 * Example Application: Pendulum Tracking
 * =======================================
 * State: x = [θ, θ̇]' (angle and angular velocity)
 * Dynamics: θ̈ = -g/L * sin(θ) - b*θ̇
 * Measurement: z = θ (we measure the angle)
 */

#ifndef EKF_EXTENDED_KALMAN_FILTER_HPP
#define EKF_EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <functional>
#include <cmath>

namespace ekf {

/**
 * @class ExtendedKalmanFilter
 * @brief Extended Kalman Filter for nonlinear systems
 *
 * The EKF uses first-order Taylor series expansion to linearize
 * the nonlinear functions around the current state estimate.
 *
 * Advantages:
 * - Handles nonlinear systems
 * - Computationally efficient
 * - Well-understood mathematically
 *
 * Limitations:
 * - Only first-order accurate (ignores higher-order terms)
 * - Can diverge for highly nonlinear systems
 * - Requires analytical Jacobian computation
 */
class ExtendedKalmanFilter {
public:
    // Function types for nonlinear dynamics and measurement
    using StateFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;
    using MeasurementFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    using JacobianFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&, double)>;
    using MeasurementJacobianFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

    /**
     * @brief Constructor
     * @param state_dim Dimension of the state vector
     * @param meas_dim Dimension of the measurement vector
     */
    ExtendedKalmanFilter(int state_dim, int meas_dim)
        : state_dim_(state_dim), meas_dim_(meas_dim)
    {
        x_ = Eigen::VectorXd::Zero(state_dim_);
        P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        Q_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        R_ = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_);
    }

    // =========================================================================
    // Setters
    // =========================================================================

    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

    /**
     * @brief Set the nonlinear state transition function
     * @param f Function x_next = f(x, dt)
     */
    void setStateFunction(StateFunction f) { f_ = f; }

    /**
     * @brief Set the state transition Jacobian
     * @param F Function that computes ∂f/∂x
     */
    void setStateJacobian(JacobianFunction F) { F_jacobian_ = F; }

    /**
     * @brief Set the nonlinear measurement function
     * @param h Function z = h(x)
     */
    void setMeasurementFunction(MeasurementFunction h) { h_ = h; }

    /**
     * @brief Set the measurement Jacobian
     * @param H Function that computes ∂h/∂x
     */
    void setMeasurementJacobian(MeasurementJacobianFunction H) { H_jacobian_ = H; }

    // =========================================================================
    // Getters
    // =========================================================================

    const Eigen::VectorXd& getState() const { return x_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }

    // =========================================================================
    // Core Filter Operations
    // =========================================================================

    /**
     * @brief Prediction Step - Propagate through nonlinear dynamics
     *
     * EKF Prediction:
     * 1. Propagate state through nonlinear function: x̂⁻ = f(x̂, dt)
     * 2. Linearize around current estimate: F = ∂f/∂x |_{x=x̂}
     * 3. Propagate covariance: P⁻ = F*P*F' + Q
     *
     * Note: The covariance propagation uses the linearized system,
     * which is an approximation. This is where EKF can lose accuracy
     * for highly nonlinear systems.
     *
     * @param dt Time step
     */
    void predict(double dt)
    {
        // Step 1: Propagate state through nonlinear dynamics
        // This captures the full nonlinear behavior for the mean
        x_ = f_(x_, dt);

        // Step 2: Compute Jacobian at current state
        // This linearizes the system for covariance propagation
        Eigen::MatrixXd F = F_jacobian_(x_, dt);

        // Step 3: Propagate covariance using linearized system
        // P⁻ = F*P*F' + Q
        // This is where approximation occurs - we're treating the
        // nonlinear system as locally linear
        P_ = F * P_ * F.transpose() + Q_;
    }

    /**
     * @brief Update Step - Incorporate measurement
     *
     * EKF Update:
     * 1. Compute predicted measurement: ẑ = h(x̂⁻)
     * 2. Compute innovation: y = z - ẑ
     * 3. Linearize measurement: H = ∂h/∂x |_{x=x̂⁻}
     * 4. Compute Kalman gain: K = P⁻*H'*(H*P⁻*H' + R)⁻¹
     * 5. Update state: x̂ = x̂⁻ + K*y
     * 6. Update covariance: P = (I - K*H)*P⁻
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // Step 1 & 2: Compute innovation using nonlinear measurement
        Eigen::VectorXd z_pred = h_(x_);
        Eigen::VectorXd y = z - z_pred;

        // Step 3: Linearize measurement function
        Eigen::MatrixXd H = H_jacobian_(x_);

        // Step 4: Compute Kalman gain
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

        // Step 5: Update state
        x_ = x_ + K * y;

        // Step 6: Update covariance
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H) * P_;
    }

private:
    int state_dim_;
    int meas_dim_;

    Eigen::VectorXd x_;  ///< State estimate
    Eigen::MatrixXd P_;  ///< State covariance
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd R_;  ///< Measurement noise covariance

    StateFunction f_;                      ///< Nonlinear state transition
    JacobianFunction F_jacobian_;          ///< State transition Jacobian
    MeasurementFunction h_;                ///< Nonlinear measurement function
    MeasurementJacobianFunction H_jacobian_;  ///< Measurement Jacobian
};

// =============================================================================
// Example: Pendulum System Helper Functions
// =============================================================================

namespace pendulum {

/**
 * @brief Pendulum state transition function
 *
 * State: x = [θ, θ̇]'
 * Dynamics: θ̈ = -(g/L)*sin(θ) - b*θ̇
 *
 * Using Euler integration:
 *   θ_new = θ + θ̇*dt
 *   θ̇_new = θ̇ + θ̈*dt
 */
inline Eigen::VectorXd stateTransition(const Eigen::VectorXd& x, double dt,
                                       double g = 9.81, double L = 1.0, double b = 0.1)
{
    double theta = x(0);
    double theta_dot = x(1);

    // Angular acceleration
    double theta_ddot = -(g / L) * std::sin(theta) - b * theta_dot;

    // Euler integration
    Eigen::VectorXd x_new(2);
    x_new(0) = theta + theta_dot * dt;
    x_new(1) = theta_dot + theta_ddot * dt;

    return x_new;
}

/**
 * @brief State transition Jacobian for pendulum
 *
 * F = ∂f/∂x = [∂θ_new/∂θ    ∂θ_new/∂θ̇  ]
 *              [∂θ̇_new/∂θ    ∂θ̇_new/∂θ̇  ]
 *
 *   = [1                           dt        ]
 *     [-(g/L)*cos(θ)*dt    1 - b*dt]
 */
inline Eigen::MatrixXd stateJacobian(const Eigen::VectorXd& x, double dt,
                                     double g = 9.81, double L = 1.0, double b = 0.1)
{
    double theta = x(0);

    Eigen::MatrixXd F(2, 2);
    F(0, 0) = 1.0;
    F(0, 1) = dt;
    F(1, 0) = -(g / L) * std::cos(theta) * dt;
    F(1, 1) = 1.0 - b * dt;

    return F;
}

/**
 * @brief Measurement function (observe angle)
 */
inline Eigen::VectorXd measurementFunction(const Eigen::VectorXd& x)
{
    Eigen::VectorXd z(1);
    z(0) = x(0);  // We measure θ
    return z;
}

/**
 * @brief Measurement Jacobian
 * H = [1, 0]
 */
inline Eigen::MatrixXd measurementJacobian(const Eigen::VectorXd& /*x*/)
{
    Eigen::MatrixXd H(1, 2);
    H << 1.0, 0.0;
    return H;
}

} // namespace pendulum

} // namespace ekf

#endif // EKF_EXTENDED_KALMAN_FILTER_HPP
