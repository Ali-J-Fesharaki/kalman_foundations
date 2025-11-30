/**
 * @file KalmanFilter.hpp
 * @brief Kalman Filter Implementation - Optimal Observer (Control Theory) Perspective
 *
 * This implementation derives the Kalman Filter from the perspective of
 * Control Theory and Optimal State Estimation. This is the form commonly
 * used in robotics frameworks like Nav2/ROS.
 *
 * Key Insight: The Kalman Filter is an optimal state observer that minimizes
 * the estimation error covariance. The Kalman gain K is chosen to make the
 * error dynamics stable and optimal.
 *
 * Control Theory Framework:
 * =========================
 * System:    x[k+1] = F*x[k] + w[k]     (state dynamics)
 * Observer:  x̂[k+1] = F*x̂[k] + K*(z[k] - H*x̂[k])  (estimation dynamics)
 *
 * The goal is to design K such that the estimation error e[k] = x[k] - x̂[k]
 * converges to zero as quickly as possible while remaining stable.
 *
 * The Riccati equation provides the optimal K.
 */

#ifndef OPTIMAL_OBSERVER_KALMAN_FILTER_HPP
#define OPTIMAL_OBSERVER_KALMAN_FILTER_HPP

#include <Eigen/Dense>

namespace optimal_observer {

/**
 * @class KalmanFilter
 * @brief Linear Kalman Filter derived from control theory / observer design
 *
 * Observer Structure:
 * ===================
 * The Kalman Filter is a Luenberger observer with optimal gain K.
 *
 * Standard Luenberger observer: x̂_dot = A*x̂ + B*u + L*(y - C*x̂)
 * Kalman filter (discrete):     x̂[k|k] = x̂[k|k-1] + K*(z[k] - H*x̂[k|k-1])
 *
 * The key insight is that K (observer gain) affects:
 * 1. Error Dynamics: How fast estimation error decays
 * 2. Noise Sensitivity: How much measurement noise affects the estimate
 * 3. Stability: Whether the observer is stable
 *
 * The Kalman Filter chooses K optimally using the Riccati equation.
 */
class KalmanFilter {
public:
    /**
     * @brief Constructor initializing filter dimensions and matrices
     * @param state_dim Dimension of the state vector
     * @param meas_dim Dimension of the measurement vector
     */
    KalmanFilter(int state_dim, int meas_dim)
        : state_dim_(state_dim), meas_dim_(meas_dim)
    {
        x_ = Eigen::VectorXd::Zero(state_dim_);
        P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

        F_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        Q_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        H_ = Eigen::MatrixXd::Zero(meas_dim_, state_dim_);
        R_ = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_);
    }

    // =========================================================================
    // Setters for System Matrices
    // =========================================================================

    void setStateTransition(const Eigen::MatrixXd& F) { F_ = F; }
    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementMatrix(const Eigen::MatrixXd& H) { H_ = H; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }

    // =========================================================================
    // Getters
    // =========================================================================

    const Eigen::VectorXd& getState() const { return x_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }

    // =========================================================================
    // Core Filter Operations
    // =========================================================================

    /**
     * @brief Prediction Step - Open-Loop State Propagation
     *
     * From a control theory perspective, the prediction step is the
     * open-loop propagation of the system dynamics (no feedback from
     * measurements).
     *
     * State Propagation:
     *   x̂[k|k-1] = F * x̂[k-1|k-1]
     *
     * Error Covariance Propagation (Riccati Time Update):
     *   P[k|k-1] = F * P[k-1|k-1] * F' + Q
     *
     * This represents the uncertainty growth when the system evolves
     * without measurement correction.
     */
    void predict()
    {
        // OPEN-LOOP DYNAMICS:
        // Without measurements, we can only predict based on our model.
        // The state estimate evolves according to the system dynamics.
        x_ = F_ * x_;

        // ERROR COVARIANCE TIME UPDATE:
        // The prediction error covariance satisfies the discrete Lyapunov equation:
        //   P_pred = F * P * F' + Q
        //
        // This is the "time update" portion of the Riccati equation.
        // Q represents the process noise that excites the system.
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /**
     * @brief Update Step - Optimal Observer Correction
     *
     * ERROR DYNAMICS ANALYSIS:
     * ========================
     * Define estimation error: e[k] = x[k] - x̂[k]
     *
     * After prediction: e_pred = x - x̂_pred
     * After update:     e_new = x - x̂_new = x - (x̂_pred + K*(z - H*x̂_pred))
     *                        = x - x̂_pred - K*(H*x + v - H*x̂_pred)
     *                        = (I - K*H)*e_pred - K*v
     *
     * where v is measurement noise.
     *
     * The error dynamics are:
     *   e[k+1] = (I - K*H) * F * e[k] + noise terms
     *
     * STABILITY ANALYSIS:
     * ===================
     * The observer is stable if eigenvalues of (I - K*H)*F are inside unit circle.
     * The Kalman gain K is designed to ensure:
     * 1. Stability of error dynamics
     * 2. Minimum error covariance
     *
     * FEEDBACK GAIN K (From Riccati Equation):
     * ========================================
     * The optimal K minimizes E[e*e'] subject to the error dynamics.
     * This is found by solving the Discrete Algebraic Riccati Equation (DARE):
     *
     *   P = F*P*F' + Q - F*P*H'*(H*P*H' + R)^(-1)*H*P*F'
     *
     * The optimal gain is:
     *   K = P*H'*(H*P*H' + R)^(-1)
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // INNOVATION (Output Error):
        // ==========================
        // This is the difference between measured output and predicted output.
        // In control theory, this is the "output error" that drives the observer.
        Eigen::VectorXd innovation = z - H_ * x_;

        // INNOVATION COVARIANCE:
        // ======================
        // S = H*P*H' + R
        // This represents the expected variance of the innovation signal.
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

        // OPTIMAL FEEDBACK GAIN K:
        // ========================
        // The Kalman gain is the optimal observer gain that:
        // 1. Makes error dynamics stable: eig((I-K*H)*F) inside unit circle
        // 2. Minimizes steady-state error covariance
        //
        // K = P * H' * S^(-1)
        //
        // Interpretation in control terms:
        // - K is high when P is large (uncertain state → aggressive correction)
        // - K is low when R is large (noisy sensor → trust prediction more)
        //
        // This is the dual of LQR: LQR finds optimal control gain,
        // Kalman finds optimal observer gain.
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        // STATE CORRECTION (Observer Update):
        // ====================================
        // x̂[k|k] = x̂[k|k-1] + K * innovation
        //
        // This is the Luenberger observer update with optimal gain K.
        // The innovation drives the estimate toward the true state.
        x_ = x_ + K * innovation;

        // ERROR COVARIANCE MEASUREMENT UPDATE:
        // ====================================
        // P[k|k] = (I - K*H) * P[k|k-1]
        //
        // This is the "measurement update" of the Riccati equation.
        // The covariance decreases because measurement information
        // reduces our uncertainty about the state.
        //
        // In steady state, this converges to the solution of the DARE.
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H_) * P_;

        // STABILITY NOTE:
        // ===============
        // The closed-loop error dynamics matrix is (I - K*H)*F.
        // With optimal K from Riccati equation, this is guaranteed stable
        // if (F, H) is observable and (F, sqrt(Q)) is controllable.
        // These are the standard detectability/stabilizability conditions.
    }

private:
    int state_dim_;  ///< Dimension of state vector (system order)
    int meas_dim_;   ///< Dimension of measurement vector (number of sensors)

    Eigen::VectorXd x_;  ///< State estimate (observer state)
    Eigen::MatrixXd P_;  ///< Error covariance (Riccati variable)

    Eigen::MatrixXd F_;  ///< State transition matrix (system A matrix)
    Eigen::MatrixXd Q_;  ///< Process noise covariance (disturbance model)
    Eigen::MatrixXd H_;  ///< Measurement matrix (system C matrix)
    Eigen::MatrixXd R_;  ///< Measurement noise covariance (sensor model)
};

} // namespace optimal_observer

#endif // OPTIMAL_OBSERVER_KALMAN_FILTER_HPP
