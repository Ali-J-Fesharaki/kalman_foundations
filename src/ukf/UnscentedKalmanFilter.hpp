/**
 * @file UnscentedKalmanFilter.hpp
 * @brief Unscented Kalman Filter (UKF) Implementation
 *
 * The Unscented Kalman Filter handles nonlinear systems using the Unscented
 * Transform, which captures the mean and covariance to the 3rd order (Taylor series)
 * for any nonlinearity.
 *
 * Mathematical Foundation:
 * ========================
 * Instead of linearizing, UKF uses sigma points to capture the probability
 * distribution. These points are propagated through the nonlinear functions,
 * then used to recover the transformed mean and covariance.
 *
 * Sigma Points:
 * For n-dimensional state with covariance P:
 *   χ₀ = x̄                           (mean)
 *   χᵢ = x̄ + (√((n+λ)P))ᵢ    for i = 1,...,n
 *   χᵢ = x̄ - (√((n+λ)P))ᵢ₋ₙ  for i = n+1,...,2n
 *
 * where λ = α²(n+κ) - n is the scaling parameter
 *
 * Advantages over EKF:
 * - No Jacobian computation required
 * - More accurate for highly nonlinear systems
 * - Captures mean and covariance to 3rd order
 */

#ifndef UKF_UNSCENTED_KALMAN_FILTER_HPP
#define UKF_UNSCENTED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <vector>

namespace ukf {

/**
 * @class UnscentedKalmanFilter
 * @brief Unscented Kalman Filter for nonlinear systems
 *
 * The UKF uses the Unscented Transform to propagate probability distributions
 * through nonlinear functions, avoiding the need for Jacobian computation.
 */
class UnscentedKalmanFilter {
public:
    using StateFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;
    using MeasurementFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    /**
     * @brief Constructor
     * @param state_dim Dimension of the state vector
     * @param meas_dim Dimension of the measurement vector
     * @param alpha Spread of sigma points (typically 1e-3)
     * @param beta Prior knowledge of distribution (2 for Gaussian)
     * @param kappa Secondary scaling parameter (typically 0 or 3-n)
     */
    UnscentedKalmanFilter(int state_dim, int meas_dim,
                          double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : state_dim_(state_dim), meas_dim_(meas_dim),
          alpha_(alpha), beta_(beta), kappa_(kappa)
    {
        x_ = Eigen::VectorXd::Zero(state_dim_);
        P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        Q_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        R_ = Eigen::MatrixXd::Identity(meas_dim_, meas_dim_);

        // Compute UKF parameters
        computeWeights();
    }

    // =========================================================================
    // Setters
    // =========================================================================

    void setProcessNoise(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setMeasurementNoise(const Eigen::MatrixXd& R) { R_ = R; }
    void setState(const Eigen::VectorXd& x) { x_ = x; }
    void setCovariance(const Eigen::MatrixXd& P) { P_ = P; }
    void setStateFunction(StateFunction f) { f_ = f; }
    void setMeasurementFunction(MeasurementFunction h) { h_ = h; }

    // =========================================================================
    // Getters
    // =========================================================================

    const Eigen::VectorXd& getState() const { return x_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }

    // =========================================================================
    // Core Filter Operations
    // =========================================================================

    /**
     * @brief Prediction Step using Unscented Transform
     *
     * UKF Prediction:
     * 1. Generate sigma points from current state and covariance
     * 2. Propagate each sigma point through nonlinear dynamics
     * 3. Compute predicted mean and covariance from propagated points
     *
     * @param dt Time step
     */
    void predict(double dt)
    {
        // Step 1: Generate sigma points
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);

        // Step 2: Propagate sigma points through nonlinear dynamics
        std::vector<Eigen::VectorXd> propagated_points(2 * state_dim_ + 1);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            propagated_points[i] = f_(sigma_points[i], dt);
        }

        // Step 3: Compute predicted mean
        x_ = Eigen::VectorXd::Zero(state_dim_);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            x_ += Wm_[i] * propagated_points[i];
        }

        // Step 4: Compute predicted covariance
        P_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd dx = propagated_points[i] - x_;
            P_ += Wc_[i] * dx * dx.transpose();
        }
        P_ += Q_;  // Add process noise
    }

    /**
     * @brief Update Step using Unscented Transform
     *
     * UKF Update:
     * 1. Generate sigma points from predicted state
     * 2. Transform sigma points through measurement function
     * 3. Compute predicted measurement mean and covariance
     * 4. Compute cross-covariance between state and measurement
     * 5. Compute Kalman gain and update state
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // Step 1: Generate sigma points from predicted state
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);

        // Step 2: Transform sigma points through measurement function
        std::vector<Eigen::VectorXd> meas_points(2 * state_dim_ + 1);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            meas_points[i] = h_(sigma_points[i]);
        }

        // Step 3: Compute predicted measurement mean
        Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(meas_dim_);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            z_pred += Wm_[i] * meas_points[i];
        }

        // Step 4: Compute measurement covariance Pzz
        Eigen::MatrixXd Pzz = Eigen::MatrixXd::Zero(meas_dim_, meas_dim_);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd dz = meas_points[i] - z_pred;
            Pzz += Wc_[i] * dz * dz.transpose();
        }
        Pzz += R_;  // Add measurement noise

        // Step 5: Compute cross-covariance Pxz
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(state_dim_, meas_dim_);
        for (int i = 0; i <= 2 * state_dim_; ++i) {
            Eigen::VectorXd dx = sigma_points[i] - x_;
            Eigen::VectorXd dz = meas_points[i] - z_pred;
            Pxz += Wc_[i] * dx * dz.transpose();
        }

        // Step 6: Compute Kalman gain
        Eigen::MatrixXd K = Pxz * Pzz.inverse();

        // Step 7: Update state and covariance
        x_ = x_ + K * (z - z_pred);
        P_ = P_ - K * Pzz * K.transpose();
    }

private:
    int state_dim_;
    int meas_dim_;

    // UKF tuning parameters
    double alpha_;  ///< Spread of sigma points
    double beta_;   ///< Prior knowledge (2 for Gaussian)
    double kappa_;  ///< Secondary scaling parameter
    double lambda_; ///< Composite scaling parameter

    // Weights for sigma points
    std::vector<double> Wm_;  ///< Weights for mean
    std::vector<double> Wc_;  ///< Weights for covariance

    Eigen::VectorXd x_;  ///< State estimate
    Eigen::MatrixXd P_;  ///< State covariance
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd R_;  ///< Measurement noise covariance

    StateFunction f_;        ///< Nonlinear state transition
    MeasurementFunction h_;  ///< Nonlinear measurement function

    /**
     * @brief Compute sigma point weights
     */
    void computeWeights()
    {
        lambda_ = alpha_ * alpha_ * (state_dim_ + kappa_) - state_dim_;

        // Resize weight vectors
        Wm_.resize(2 * state_dim_ + 1);
        Wc_.resize(2 * state_dim_ + 1);

        // Weight for mean (center sigma point)
        Wm_[0] = lambda_ / (state_dim_ + lambda_);
        Wc_[0] = Wm_[0] + (1 - alpha_ * alpha_ + beta_);

        // Weights for other sigma points
        for (int i = 1; i <= 2 * state_dim_; ++i) {
            Wm_[i] = 1.0 / (2.0 * (state_dim_ + lambda_));
            Wc_[i] = Wm_[i];
        }
    }

    /**
     * @brief Generate sigma points
     * @param x Mean state
     * @param P Covariance
     * @return Vector of sigma points
     */
    std::vector<Eigen::VectorXd> generateSigmaPoints(const Eigen::VectorXd& x,
                                                     const Eigen::MatrixXd& P)
    {
        std::vector<Eigen::VectorXd> sigma_points(2 * state_dim_ + 1);

        // Compute matrix square root: sqrt((n + λ) * P)
        double scale = std::sqrt(state_dim_ + lambda_);
        Eigen::LLT<Eigen::MatrixXd> llt(P);
        Eigen::MatrixXd L = llt.matrixL();
        L *= scale;

        // Center sigma point
        sigma_points[0] = x;

        // Positive and negative sigma points
        for (int i = 0; i < state_dim_; ++i) {
            sigma_points[i + 1] = x + L.col(i);
            sigma_points[i + 1 + state_dim_] = x - L.col(i);
        }

        return sigma_points;
    }
};

// =============================================================================
// Example: Pendulum System Helper Functions (same as EKF for comparison)
// =============================================================================

namespace pendulum {

/**
 * @brief Pendulum state transition function
 * State: x = [θ, θ̇]'
 * Dynamics: θ̈ = -(g/L)*sin(θ) - b*θ̇
 */
inline Eigen::VectorXd stateTransition(const Eigen::VectorXd& x, double dt,
                                       double g = 9.81, double L = 1.0, double b = 0.1)
{
    double theta = x(0);
    double theta_dot = x(1);

    double theta_ddot = -(g / L) * std::sin(theta) - b * theta_dot;

    Eigen::VectorXd x_new(2);
    x_new(0) = theta + theta_dot * dt;
    x_new(1) = theta_dot + theta_ddot * dt;

    return x_new;
}

/**
 * @brief Measurement function (observe angle)
 */
inline Eigen::VectorXd measurementFunction(const Eigen::VectorXd& x)
{
    Eigen::VectorXd z(1);
    z(0) = x(0);
    return z;
}

} // namespace pendulum

} // namespace ukf

#endif // UKF_UNSCENTED_KALMAN_FILTER_HPP
