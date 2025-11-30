/**
 * @file KalmanFilter.hpp
 * @brief Kalman Filter Implementation - Geometric (Projection) Perspective
 *
 * This implementation derives the Kalman Filter from the perspective of
 * Orthogonal Projection in Hilbert Spaces.
 *
 * Key Insight: The Kalman Filter computes the orthogonal projection of the
 * state vector onto the linear subspace spanned by the observations.
 *
 * Mathematical Foundation:
 * ========================
 * In a Hilbert space of random variables with inner product:
 *   <X, Y> = E[XY']
 *
 * The optimal estimate is the projection of the true state onto the
 * observation space. The innovation (error) is orthogonal to all observations.
 *
 * Orthogonality Principle:
 *   E[(x - x̂)(z - ẑ)'] = 0
 *
 * This orthogonality condition directly leads to the Kalman gain formula.
 */

#ifndef GEOMETRIC_KALMAN_FILTER_HPP
#define GEOMETRIC_KALMAN_FILTER_HPP

#include <Eigen/Dense>

namespace geometric {

/**
 * @class KalmanFilter
 * @brief Linear Kalman Filter derived from geometric projection theory
 *
 * Hilbert Space Framework:
 * ========================
 * Consider the Hilbert space L² of square-integrable random variables.
 * - Inner product: <X, Y> = E[XY']
 * - Norm: ||X|| = sqrt(E[X'X])
 *
 * The optimal linear estimator is the orthogonal projection of x onto
 * the subspace spanned by {1, z₁, z₂, ..., z_k} where z_i are observations.
 *
 * The Kalman Filter implements this projection recursively, avoiding the
 * need to store all past observations.
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
        // State estimate (projected state in Hilbert space)
        x_ = Eigen::VectorXd::Zero(state_dim_);

        // Covariance (measure of estimation error in the space)
        P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

        // System matrices
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
     * @brief Prediction Step - Linear Transformation in Hilbert Space
     *
     * The prediction step is a linear transformation of the state vector
     * in the Hilbert space. Under linear transformations:
     *   - Projected estimate transforms: x̂_pred = F * x̂
     *   - Error covariance transforms: P_pred = F*P*F' + Q
     *
     * Geometrically, F rotates and scales the subspace spanned by
     * our current estimate, while Q adds dimensions corresponding
     * to process noise uncertainty.
     */
    void predict()
    {
        // LINEAR TRANSFORMATION IN HILBERT SPACE:
        // The prediction step applies a linear operator F to the
        // current projected estimate.

        // Transform the projected state
        x_ = F_ * x_;

        // Transform the error covariance ellipsoid
        // Q expands the subspace to account for model uncertainty
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /**
     * @brief Update Step - Orthogonal Projection onto Observation Space
     *
     * This is the geometric heart of the Kalman Filter. We project the
     * predicted state onto the affine subspace defined by the new measurement.
     *
     * ORTHOGONALITY PRINCIPLE:
     * ========================
     * The optimal estimate x̂ satisfies:
     *   E[(x - x̂) * v'] = 0  for all linear combinations v of observations
     *
     * This means the estimation error is orthogonal (in the L² sense) to
     * the observation space. The innovation ν = z - H*x̂ must be orthogonal
     * to the prediction error.
     *
     * PROJECTION OPERATOR:
     * ====================
     * The Kalman gain K acts as a projection operator that maps the
     * innovation from observation space back to state space:
     *   K = P * H' * (H*P*H' + R)^(-1)
     *
     * This formula comes directly from the orthogonality condition:
     *   E[(x - x̂_new)(z - H*x̂_pred)'] = 0
     *
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z)
    {
        // COMPUTE INNOVATION (Projection Residual)
        // =========================================
        // The innovation ν is the difference between the measurement
        // and its projection onto our current estimate subspace.
        //
        // ν = z - H*x̂_pred
        //
        // In Hilbert space terms, this represents the component of z
        // that lies outside our current estimation subspace.
        Eigen::VectorXd innovation = z - H_ * x_;

        // INNOVATION COVARIANCE (Inner Product Structure)
        // ================================================
        // S = E[ν * ν'] = H*P*H' + R
        //
        // This is the Gram matrix of the innovation, encoding the
        // inner product structure in the observation space.
        // - H*P*H': contribution from prediction uncertainty
        // - R: contribution from measurement noise
        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

        // PROJECTION OPERATOR (Kalman Gain)
        // ==================================
        // The Kalman gain K is derived from the orthogonality principle.
        //
        // We seek K such that: x̂_new = x̂_pred + K*ν
        // with the constraint that estimation error is orthogonal to z.
        //
        // E[(x - x̂_new) * z'] = 0
        // E[(x - x̂_pred - K*ν) * z'] = 0
        //
        // Solving this orthogonality condition yields:
        // K = E[(x - x̂_pred) * ν'] * E[ν * ν']^(-1)
        // K = P * H' * S^(-1)
        //
        // Geometrically, K projects from observation space back to state space,
        // scaled appropriately by the relative uncertainties.
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        // STATE UPDATE (Projection onto New Subspace)
        // ============================================
        // The new estimate is the orthogonal projection of x onto the
        // enlarged subspace that now includes the new measurement z.
        //
        // x̂_new = x̂_pred + K * ν
        //
        // This can be interpreted as:
        // "Old estimate" + "Correction from new information"
        // where the correction is the innovation projected back to state space.
        x_ = x_ + K * innovation;

        // COVARIANCE UPDATE (Error Subspace Reduction)
        // =============================================
        // After projection, the estimation error is reduced.
        // The new error subspace is orthogonal to all observations.
        //
        // P_new = (I - K*H) * P_pred
        //
        // Geometrically, the error covariance ellipsoid shrinks in the
        // directions that are observable through H.
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H_) * P_;

        // VERIFICATION: The innovation is orthogonal to the updated error
        // ================================================================
        // After update, we have: E[(x - x̂_new) * ν'] = 0
        // This is the fundamental orthogonality that defines optimal estimation.
        // (Verification is implicit in the derivation above)
    }

private:
    int state_dim_;  ///< Dimension of state vector
    int meas_dim_;   ///< Dimension of measurement vector

    Eigen::VectorXd x_;  ///< State estimate (projection onto observation space)
    Eigen::MatrixXd P_;  ///< Error covariance (geometry of error subspace)

    Eigen::MatrixXd F_;  ///< State transition (linear operator)
    Eigen::MatrixXd Q_;  ///< Process noise covariance
    Eigen::MatrixXd H_;  ///< Measurement matrix (observation operator)
    Eigen::MatrixXd R_;  ///< Measurement noise covariance
};

} // namespace geometric

#endif // GEOMETRIC_KALMAN_FILTER_HPP
