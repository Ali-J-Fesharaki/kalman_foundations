/**
 * @file KalmanFilter.hpp
 * @brief Kalman Filter Implementation - Statistical (MMSE) Perspective
 *
 * This implementation derives the Kalman Filter from the perspective of
 * Minimum Mean Squared Error (MMSE) estimation and Weighted Least Squares.
 *
 * Key Insight: The Kalman Filter minimizes a quadratic cost function that
 * penalizes both deviation from the prediction and deviation from measurements,
 * weighted by their respective uncertainties.
 *
 * Cost Function:
 * ==============
 * J(x) = (x - x_pred)' * P_pred^(-1) * (x - x_pred)    [prediction term]
 *      + (z - H*x)' * R^(-1) * (z - H*x)               [measurement term]
 *
 * The optimal x minimizes J(x), which is a Weighted Least Squares problem.
 */

#ifndef STATISTICAL_KALMAN_FILTER_HPP
#define STATISTICAL_KALMAN_FILTER_HPP

#include <Eigen/Dense>

namespace statistical {

/**
 * @class KalmanFilter
 * @brief Linear Kalman Filter derived from MMSE / Weighted Least Squares
 *
 * Mathematical Foundation:
 * ========================
 * We seek the estimator x̂ that minimizes the Mean Squared Error:
 *   MSE = E[||x - x̂||²]
 *
 * For Gaussian distributions, the MMSE estimator equals the conditional mean,
 * and can be found by solving a Weighted Least Squares problem.
 *
 * The WLS problem balances two competing objectives:
 * 1. Stay close to the prior estimate (weighted by P^(-1))
 * 2. Explain the measurements well (weighted by R^(-1))
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
     * @brief Prediction Step - Propagate Prior Statistics
     *
     * From the MMSE perspective, prediction propagates the sufficient
     * statistics (mean and covariance) through the system dynamics.
     *
     * This creates the prior for the next optimization problem.
     */
    void predict()
    {
        // Propagate mean: x_pred = F * x
        x_ = F_ * x_;

        // Propagate covariance: P_pred = F*P*F' + Q
        // Q represents uncertainty in the dynamics model
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /**
     * @brief Update Step - Solve Weighted Least Squares Problem
     *
     * COST FUNCTION TO MINIMIZE:
     * ==========================
     * J(x) = (x - x_pred)' * P^(-1) * (x - x_pred)     ... (1) Prior deviation
     *      + (z - H*x)' * R^(-1) * (z - H*x)           ... (2) Measurement deviation
     *
     * Term (1): Penalizes deviation from prediction, weighted by inverse covariance.
     *           Small P → high confidence → large penalty for deviation.
     *
     * Term (2): Penalizes deviation from measurements, weighted by inverse noise cov.
     *           Small R → precise measurements → large penalty for deviation.
     *
     * SOLVING THE OPTIMIZATION:
     * =========================
     * Taking gradient and setting to zero:
     *   ∂J/∂x = 2*P^(-1)*(x - x_pred) - 2*H'*R^(-1)*(z - H*x) = 0
     *
     * Rearranging:
     *   (P^(-1) + H'*R^(-1)*H) * x = P^(-1)*x_pred + H'*R^(-1)*z
     *
     * This is a linear system Ax = b with:
     *   A = P^(-1) + H'*R^(-1)*H  (Information matrix)
     *   b = P^(-1)*x_pred + H'*R^(-1)*z  (Information vector)
     *
     * The solution x* = A^(-1)*b can be rewritten using matrix identities
     * to get the standard Kalman filter form.
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // =====================================================================
        // WEIGHTED LEAST SQUARES INTERPRETATION
        // =====================================================================
        //
        // We are solving the optimization problem:
        //   minimize_x  J(x) = ||x - x_pred||²_P + ||z - H*x||²_R
        //
        // where ||v||²_W = v' * W^(-1) * v is the weighted norm.
        //
        // This is equivalent to finding the point that optimally balances
        // our prior belief with the new evidence from measurements.

        // INNOVATION: How far is measurement from expected value?
        // This represents the "residual" in the least squares sense.
        Eigen::VectorXd y = z - H_ * x_;

        // INNOVATION COVARIANCE: Total uncertainty in the residual
        // S combines prediction uncertainty (mapped to observation space)
        // with measurement noise.
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

        // KALMAN GAIN: Optimal weighting factor
        // =====================================
        // K determines how to weight the innovation when correcting the estimate.
        //
        // From the WLS solution:
        //   x* = (P^(-1) + H'*R^(-1)*H)^(-1) * (P^(-1)*x_pred + H'*R^(-1)*z)
        //
        // Using the matrix inversion lemma (Woodbury identity):
        //   (P^(-1) + H'*R^(-1)*H)^(-1) = P - P*H'*(H*P*H' + R)^(-1)*H*P
        //
        // After algebraic manipulation, the update becomes:
        //   x* = x_pred + K*(z - H*x_pred)
        //
        // where K = P*H'*S^(-1)
        //
        // Interpretation of K:
        // - K weights the innovation optimally based on relative uncertainties
        // - Large P (uncertain prediction) → Large K → Trust measurement more
        // - Large R (noisy measurement) → Small K → Trust prediction more
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        // STATE UPDATE: Apply optimal correction
        // x* minimizes the cost function J(x)
        x_ = x_ + K * y;

        // COVARIANCE UPDATE: New uncertainty after incorporating measurement
        // =================================================================
        // The posterior covariance comes from the inverse of the Hessian
        // of the cost function at the optimal point:
        //
        //   P_new = (∂²J/∂x²)^(-1) = (P^(-1) + H'*R^(-1)*H)^(-1)
        //
        // Using the matrix inversion lemma, this simplifies to:
        //   P_new = (I - K*H) * P
        //
        // The covariance always decreases because we've gained information
        // from the measurement.
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H_) * P_;
    }

private:
    int state_dim_;  ///< Dimension of state vector
    int meas_dim_;   ///< Dimension of measurement vector

    Eigen::VectorXd x_;  ///< State estimate (optimal in MMSE sense)
    Eigen::MatrixXd P_;  ///< State covariance (curvature of cost function)

    Eigen::MatrixXd F_;  ///< State transition matrix
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd H_;  ///< Measurement matrix
    Eigen::MatrixXd R_;  ///< Measurement noise covariance
};

} // namespace statistical

#endif // STATISTICAL_KALMAN_FILTER_HPP
