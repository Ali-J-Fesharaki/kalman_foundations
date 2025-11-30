/**
 * @file KalmanFilter.hpp
 * @brief Kalman Filter Implementation - Bayesian Perspective
 *
 * This implementation derives the Kalman Filter from the perspective of
 * Probability Density Functions and Bayes' Rule.
 *
 * The core idea is that both our prediction (prior) and measurement (likelihood)
 * are Gaussian distributions. The update step computes the posterior by
 * multiplying these two Gaussians together:
 *
 *   P(x|z) ∝ P(z|x) * P(x)
 *   posterior ∝ likelihood * prior
 *
 * When multiplying two Gaussians, the result is another Gaussian whose mean
 * and covariance can be computed analytically.
 */

#ifndef BAYESIAN_KALMAN_FILTER_HPP
#define BAYESIAN_KALMAN_FILTER_HPP

#include <Eigen/Dense>

namespace bayesian {

/**
 * @class KalmanFilter
 * @brief Linear Kalman Filter derived from Bayesian probability theory
 *
 * Mathematical Foundation:
 * ========================
 * Prior Distribution (prediction):      P(x) = N(x_pred, P_pred)
 * Likelihood Function (measurement):    P(z|x) = N(H*x, R)
 * Posterior Distribution (update):      P(x|z) ∝ P(z|x) * P(x)
 *
 * The product of two Gaussians N(μ₁, Σ₁) and N(μ₂, Σ₂) gives:
 *   Posterior Mean: μ = Σ * (Σ₁⁻¹*μ₁ + Σ₂⁻¹*μ₂)
 *   Posterior Cov:  Σ = (Σ₁⁻¹ + Σ₂⁻¹)⁻¹
 *
 * Applying this to prediction N(x_pred, P_pred) and measurement N(z, R):
 *   - Transform measurement to state space via H
 *   - Combine using Bayes' rule
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
        // Initialize state estimate
        x_ = Eigen::VectorXd::Zero(state_dim_);

        // Initialize covariance as identity (unit uncertainty)
        P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

        // Initialize system matrices to identity/zero defaults
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
     * @brief Prediction Step - Propagate Uncertainty Forward
     *
     * From a Bayesian perspective, this step computes the prior distribution
     * for the next time step by propagating the current posterior through
     * the system dynamics.
     *
     * Prior Propagation:
     *   x_pred = F * x      (propagate mean through linear dynamics)
     *   P_pred = F*P*F' + Q (propagate uncertainty + add process noise)
     *
     * The process noise Q represents our uncertainty about the system dynamics,
     * modeling unmodeled accelerations, disturbances, etc.
     */
    void predict()
    {
        // UNCERTAINTY PROPAGATION:
        // The prior distribution is transformed through the linear system model.
        // For a Gaussian, this means transforming both mean and covariance.

        // Propagate state mean: E[x_k+1] = F * E[x_k]
        x_ = F_ * x_;

        // Propagate covariance: Cov[x_k+1] = F * Cov[x_k] * F' + Q
        // This follows from the linear transformation of random variables:
        // If y = Ax + b, then Cov[y] = A * Cov[x] * A'
        // Q is added to account for process noise (uncertainty in dynamics)
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /**
     * @brief Update Step - Apply Bayes' Rule
     *
     * This is the heart of the Bayesian interpretation. We have:
     *   - Prior: P(x) = N(x_pred, P_pred)
     *   - Likelihood: P(z|x) = N(H*x, R)
     *
     * Using Bayes' Rule: P(x|z) ∝ P(z|x) * P(x)
     *
     * The product of two Gaussians gives the optimal posterior estimate.
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // BAYES RULE APPLICATION:
        // posterior ∝ likelihood * prior
        //
        // For Gaussian distributions, this product results in another Gaussian.
        // The formulas below are derived from completing the square in the
        // exponent of the product of two Gaussian PDFs.

        // Innovation (measurement residual):
        // This is the difference between what we measured and what we expected
        // based on our prior. In probabilistic terms, it's the "surprise" from
        // the measurement.
        Eigen::VectorXd y = z - H_ * x_;

        // Innovation Covariance:
        // This combines our prediction uncertainty (H*P*H') with measurement
        // uncertainty (R). It represents total uncertainty in the innovation.
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

        // Kalman Gain:
        // This is derived from the formula for multiplying two Gaussians.
        // It determines how much we "trust" the measurement vs the prediction.
        //
        // K = P * H' * S^(-1)
        //
        // Interpretation: K weights the innovation to produce the optimal
        // correction to our prior estimate. When R is small (precise measurements),
        // K is large and we trust measurements more. When P is small (confident
        // prediction), K is small and we trust our prediction more.
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        // Update state (posterior mean):
        // x_posterior = x_prior + K * innovation
        //
        // This is the mean of the posterior Gaussian, obtained by combining
        // prior and likelihood through Bayes' rule.
        x_ = x_ + K * y;

        // Update covariance (posterior uncertainty):
        // P_posterior = (I - K*H) * P_prior
        //
        // The posterior uncertainty is ALWAYS less than or equal to the prior
        // uncertainty. Information from measurements can only reduce uncertainty.
        // This is the essence of Bayesian updating: evidence narrows our beliefs.
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H_) * P_;
    }

private:
    int state_dim_;  ///< Dimension of state vector
    int meas_dim_;   ///< Dimension of measurement vector

    Eigen::VectorXd x_;  ///< State estimate (posterior mean)
    Eigen::MatrixXd P_;  ///< State covariance (posterior uncertainty)

    Eigen::MatrixXd F_;  ///< State transition matrix
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd H_;  ///< Measurement matrix
    Eigen::MatrixXd R_;  ///< Measurement noise covariance
};

} // namespace bayesian

#endif // BAYESIAN_KALMAN_FILTER_HPP
