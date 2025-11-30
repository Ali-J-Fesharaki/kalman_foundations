/**
 * @file main_gui.cpp
 * @brief Educational GUI for Kalman Filter Foundations
 *
 * This GUI helps users learn the four mathematical perspectives of the Kalman Filter:
 * 1. Bayesian (Probability Density Functions)
 * 2. Geometric (Orthogonal Projection)
 * 3. Statistical (Minimum Mean Squared Error)
 * 4. Optimal Observer (Control Theory)
 *
 * Features:
 * - Step-by-step execution with detailed explanations
 * - Real-time plotting of state estimates vs ground truth
 * - Visual representation of covariance ellipses
 * - Side-by-side comparison of all four perspectives
 */

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>

// Include all four Kalman Filter implementations
#include "bayesian/KalmanFilter.hpp"
#include "geometric/KalmanFilter.hpp"
#include "statistical/KalmanFilter.hpp"
#include "optimal_observer/KalmanFilter.hpp"

/**
 * @struct SimulationState
 * @brief Holds the complete state of the Kalman Filter simulation
 */
struct SimulationState {
    // System parameters (float for ImGui compatibility)
    float dt = 1.0f;
    float process_noise_intensity = 0.1f;
    float measurement_noise_variance = 1.0f;
    
    // Initial conditions (float for ImGui compatibility)
    float initial_position = 0.0f;
    float initial_velocity = 1.0f;
    float initial_covariance = 1.0f;
    
    // Simulation control
    int current_step = 0;
    int max_steps = 20;
    bool running = false;
    bool paused = false;
    float simulation_speed = 1.0f;
    
    // Data storage
    std::vector<double> time_steps;
    std::vector<double> true_positions;
    std::vector<double> measurements;
    std::vector<double> bayesian_estimates;
    std::vector<double> geometric_estimates;
    std::vector<double> statistical_estimates;
    std::vector<double> observer_estimates;
    std::vector<double> bayesian_velocities;
    std::vector<double> covariance_trace;
    
    // Filter instances
    std::unique_ptr<bayesian::KalmanFilter> kf_bayesian;
    std::unique_ptr<geometric::KalmanFilter> kf_geometric;
    std::unique_ptr<statistical::KalmanFilter> kf_statistical;
    std::unique_ptr<optimal_observer::KalmanFilter> kf_observer;
    
    // System matrices
    Eigen::MatrixXd F, H, Q, R, P0;
    Eigen::VectorXd x0, x_true;
    
    // Random number generator
    std::mt19937 rng{42};
    
    void initialize() {
        // Clear previous data
        time_steps.clear();
        true_positions.clear();
        measurements.clear();
        bayesian_estimates.clear();
        geometric_estimates.clear();
        statistical_estimates.clear();
        observer_estimates.clear();
        bayesian_velocities.clear();
        covariance_trace.clear();
        current_step = 0;
        
        const int state_dim = 2;
        const int meas_dim = 1;
        
        // State transition matrix
        F = Eigen::MatrixXd(state_dim, state_dim);
        F << 1, dt,
             0, 1;
        
        // Measurement matrix (observe position only)
        H = Eigen::MatrixXd(meas_dim, state_dim);
        H << 1, 0;
        
        // Process noise covariance
        double q = process_noise_intensity;
        Q = Eigen::MatrixXd(state_dim, state_dim);
        Q << (q * dt * dt * dt / 3), (q * dt * dt / 2),
             (q * dt * dt / 2),      (q * dt);
        
        // Measurement noise covariance
        R = Eigen::MatrixXd(meas_dim, meas_dim);
        R << measurement_noise_variance;
        
        // Initial state and covariance
        x0 = Eigen::VectorXd(state_dim);
        x0 << initial_position, initial_velocity;
        
        P0 = Eigen::MatrixXd(state_dim, state_dim);
        P0 << initial_covariance, 0,
              0, initial_covariance;
        
        // True initial state
        x_true = x0;
        
        // Initialize filters
        kf_bayesian = std::make_unique<bayesian::KalmanFilter>(state_dim, meas_dim);
        kf_geometric = std::make_unique<geometric::KalmanFilter>(state_dim, meas_dim);
        kf_statistical = std::make_unique<statistical::KalmanFilter>(state_dim, meas_dim);
        kf_observer = std::make_unique<optimal_observer::KalmanFilter>(state_dim, meas_dim);
        
        auto configureFilter = [&](auto& kf) {
            kf->setStateTransition(F);
            kf->setMeasurementMatrix(H);
            kf->setProcessNoise(Q);
            kf->setMeasurementNoise(R);
            kf->setState(x0);
            kf->setCovariance(P0);
        };
        
        configureFilter(kf_bayesian);
        configureFilter(kf_geometric);
        configureFilter(kf_statistical);
        configureFilter(kf_observer);
        
        // Reset RNG
        rng.seed(42);
    }
    
    void step() {
        if (current_step >= max_steps || !kf_bayesian) return;
        
        std::normal_distribution<double> meas_noise(0.0, std::sqrt(measurement_noise_variance));
        
        // True state evolution
        if (current_step > 0) {
            x_true = F * x_true;
        }
        
        // Generate noisy measurement
        double true_pos = x_true(0);
        double measurement = true_pos + meas_noise(rng);
        
        // Create measurement vector
        Eigen::VectorXd z(1);
        z << measurement;
        
        // Run predict step on all filters
        kf_bayesian->predict();
        kf_geometric->predict();
        kf_statistical->predict();
        kf_observer->predict();
        
        // Run update step on all filters
        kf_bayesian->update(z);
        kf_geometric->update(z);
        kf_statistical->update(z);
        kf_observer->update(z);
        
        // Store data
        time_steps.push_back(static_cast<double>(current_step));
        true_positions.push_back(true_pos);
        measurements.push_back(measurement);
        bayesian_estimates.push_back(kf_bayesian->getState()(0));
        geometric_estimates.push_back(kf_geometric->getState()(0));
        statistical_estimates.push_back(kf_statistical->getState()(0));
        observer_estimates.push_back(kf_observer->getState()(0));
        bayesian_velocities.push_back(kf_bayesian->getState()(1));
        covariance_trace.push_back(kf_bayesian->getCovariance().trace());
        
        current_step++;
    }
    
    void reset() {
        initialize();
        running = false;
        paused = false;
    }
};

/**
 * @brief Render the main window with educational content
 */
void renderMainWindow(SimulationState& state) {
    ImGui::SetNextWindowSize(ImVec2(400, 700), ImGuiCond_FirstUseEver);
    ImGui::Begin("Kalman Filter Foundations", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Title and introduction
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Welcome to the Kalman Filter Learning Tool!");
    ImGui::Separator();
    
    ImGui::TextWrapped(
        "This tool demonstrates that the Kalman Filter can be derived from "
        "four completely different mathematical perspectives, yet all produce "
        "IDENTICAL results. This is one of the beautiful aspects of optimal "
        "estimation theory.");
    
    ImGui::Spacing();
    
    // Simulation controls
    if (ImGui::CollapsingHeader("Simulation Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Initialize")) {
            state.initialize();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            state.reset();
        }
        
        ImGui::Separator();
        
        if (!state.kf_bayesian) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 
                "Click 'Initialize' to set up the simulation");
        } else {
            if (ImGui::Button(state.running ? "Pause" : "Run Auto")) {
                if (!state.running) {
                    state.running = true;
                    state.paused = false;
                } else {
                    state.paused = !state.paused;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                state.step();
            }
            
            ImGui::SliderFloat("Speed", &state.simulation_speed, 0.1f, 5.0f, "%.1fx");
            ImGui::Text("Step: %d / %d", state.current_step, state.max_steps);
            ImGui::ProgressBar(static_cast<float>(state.current_step) / state.max_steps);
        }
    }
    
    // Parameter controls
    if (ImGui::CollapsingHeader("System Parameters")) {
        ImGui::TextWrapped("Adjust these parameters to see how they affect filter performance:");
        
        bool changed = false;
        changed |= ImGui::SliderFloat("Time Step (dt)", &state.dt, 0.1f, 2.0f);
        changed |= ImGui::SliderFloat("Process Noise (q)", &state.process_noise_intensity, 0.01f, 1.0f);
        changed |= ImGui::SliderFloat("Measurement Noise (R)", &state.measurement_noise_variance, 0.1f, 5.0f);
        changed |= ImGui::SliderInt("Max Steps", &state.max_steps, 5, 50);
        
        ImGui::Separator();
        ImGui::Text("Initial Conditions:");
        changed |= ImGui::SliderFloat("Initial Position", &state.initial_position, -5.0f, 5.0f);
        changed |= ImGui::SliderFloat("Initial Velocity", &state.initial_velocity, 0.0f, 3.0f);
        changed |= ImGui::SliderFloat("Initial Uncertainty", &state.initial_covariance, 0.1f, 5.0f);
        
        if (changed && !state.running) {
            // Optionally auto-reinitialize when parameters change
        }
    }
    
    // Current state display
    if (state.kf_bayesian && state.current_step > 0) {
        if (ImGui::CollapsingHeader("Current State", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& x = state.kf_bayesian->getState();
            const auto& P = state.kf_bayesian->getCovariance();
            
            ImGui::Text("Position Estimate: %.4f", x(0));
            ImGui::Text("Velocity Estimate: %.4f", x(1));
            ImGui::Text("Position Uncertainty: %.4f", std::sqrt(P(0, 0)));
            ImGui::Text("Velocity Uncertainty: %.4f", std::sqrt(P(1, 1)));
            
            // Check if all implementations match
            bool match = true;
            if (state.bayesian_estimates.size() > 0) {
                double bay = state.bayesian_estimates.back();
                double geo = state.geometric_estimates.back();
                double stat = state.statistical_estimates.back();
                double obs = state.observer_estimates.back();
                match = std::abs(bay - geo) < 1e-10 && 
                        std::abs(bay - stat) < 1e-10 && 
                        std::abs(bay - obs) < 1e-10;
            }
            
            if (match) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 
                    "All 4 implementations MATCH!");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
                    "WARNING: Implementations differ!");
            }
        }
    }
    
    ImGui::End();
}

/**
 * @brief Render the plot window
 */
void renderPlotWindow(SimulationState& state) {
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Position Tracking");
    
    if (state.time_steps.size() > 0 && ImPlot::BeginPlot("Position vs Time", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time Step", "Position");
        
        // Plot true position
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("True Position", state.time_steps.data(), 
                         state.true_positions.data(), static_cast<int>(state.time_steps.size()));
        
        // Plot measurements as scatter
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
        ImPlot::PlotScatter("Measurements", state.time_steps.data(), 
                           state.measurements.data(), static_cast<int>(state.time_steps.size()));
        
        // Plot Bayesian estimate
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("Bayesian Estimate", state.time_steps.data(), 
                        state.bayesian_estimates.data(), static_cast<int>(state.time_steps.size()));
        
        ImPlot::EndPlot();
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
            "Initialize and run simulation to see plots");
    }
    
    ImGui::End();
}

/**
 * @brief Render the velocity plot window
 */
void renderVelocityPlot(SimulationState& state) {
    ImGui::SetNextWindowSize(ImVec2(600, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Velocity & Uncertainty");
    
    if (state.time_steps.size() > 0 && ImPlot::BeginPlot("##vel", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time Step", "Value");
        
        // Plot velocity estimate
        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.2f, 0.8f, 1.0f), 2.0f);
        ImPlot::PlotLine("Velocity Estimate", state.time_steps.data(), 
                        state.bayesian_velocities.data(), static_cast<int>(state.time_steps.size()));
        
        // Plot true velocity (constant at initial_velocity)
        std::vector<double> true_vel(state.time_steps.size(), state.initial_velocity);
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), 1.0f);
        ImPlot::PlotLine("True Velocity", state.time_steps.data(), 
                        true_vel.data(), static_cast<int>(state.time_steps.size()));
        
        // Plot covariance trace (scaled)
        std::vector<double> scaled_cov(state.covariance_trace.size());
        for (size_t i = 0; i < state.covariance_trace.size(); i++) {
            scaled_cov[i] = state.covariance_trace[i] * 0.5;  // Scale for visibility
        }
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.3f, 0.7f), 1.5f);
        ImPlot::PlotLine("Uncertainty (scaled)", state.time_steps.data(), 
                        scaled_cov.data(), static_cast<int>(state.time_steps.size()));
        
        ImPlot::EndPlot();
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
            "Run simulation to see velocity tracking");
    }
    
    ImGui::End();
}

/**
 * @brief Render educational content about the four perspectives
 */
void renderEducationalWindow() {
    ImGui::SetNextWindowSize(ImVec2(450, 500), ImGuiCond_FirstUseEver);
    ImGui::Begin("Mathematical Perspectives");
    
    // Tabs for each perspective
    if (ImGui::BeginTabBar("PerspectiveTabs")) {
        if (ImGui::BeginTabItem("Bayesian")) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Probability Density Functions");
            ImGui::Separator();
            
            ImGui::TextWrapped(
                "Key Insight: The update step is the product of two Gaussian "
                "distributions (Prior x Likelihood).\n\n"
                "Bayes' Rule:\n"
                "  P(x|z) = P(z|x) * P(x) / P(z)\n"
                "  posterior = likelihood * prior / evidence\n\n"
                "Since P(z) is a normalization constant (independent of x), we write:\n"
                "  P(x|z) is proportional to P(z|x) * P(x)\n\n"
                "When multiplying two Gaussians, the result is another Gaussian "
                "whose mean and covariance can be computed analytically.\n\n"
                "The Kalman Gain K determines how much we 'trust' the measurement "
                "vs the prediction based on their relative uncertainties.");
            
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Equations:");
            ImGui::Text("Prior: P(x) = N(x_pred, P_pred)");
            ImGui::Text("Likelihood: P(z|x) = N(H*x, R)");
            ImGui::Text("Posterior: P(x|z) = N(x_new, P_new)");
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Geometric")) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Orthogonal Projection");
            ImGui::Separator();
            
            ImGui::TextWrapped(
                "Key Insight: The optimal estimate is the projection of the state "
                "onto the observation space in Hilbert space.\n\n"
                "In L2 space with inner product <X,Y> = E[XY']:\n"
                "- The optimal estimate x_hat minimizes ||x - x_hat||^2\n"
                "- The estimation error is orthogonal to all observations\n\n"
                "Orthogonality Principle:\n"
                "  E[(x - x_hat)(z - z_hat)'] = 0\n\n"
                "This geometric interpretation shows that estimation is fundamentally "
                "about projecting onto the 'knowable' subspace.");
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Statistical")) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Minimum Mean Squared Error");
            ImGui::Separator();
            
            ImGui::TextWrapped(
                "Key Insight: The update step solves a Weighted Least Squares "
                "optimization problem.\n\n"
                "Cost Function J(x):\n"
                "  J = (x-x_pred)'*P^-1*(x-x_pred)\n"
                "    + (z-H*x)'*R^-1*(z-H*x)\n\n"
                "This balances:\n"
                "1. Deviation from prediction (weighted by P^-1)\n"
                "2. Deviation from measurement (weighted by R^-1)\n\n"
                "Taking the gradient and setting to zero gives the Kalman Filter "
                "equations directly!");
            
            ImGui::EndTabItem();
        }
        
        if (ImGui::BeginTabItem("Control Theory")) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Optimal State Observer");
            ImGui::Separator();
            
            ImGui::TextWrapped(
                "Key Insight: The Kalman gain is the optimal observer gain from "
                "the Riccati equation.\n\n"
                "Observer Structure:\n"
                "  x_hat[k+1] = F*x_hat[k] + K*(z[k] - H*x_hat[k])\n\n"
                "The gain K affects:\n"
                "1. Error Dynamics: How fast error decays\n"
                "2. Noise Sensitivity: How much noise affects estimate\n"
                "3. Stability: Whether observer is stable\n\n"
                "The Kalman Filter is the DUAL of LQR:\n"
                "- LQR finds optimal control gain\n"
                "- Kalman finds optimal observer gain");
            
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
    
    ImGui::End();
}

/**
 * @brief Render comparison window showing all four implementations
 */
void renderComparisonWindow(SimulationState& state) {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Implementation Comparison");
    
    if (state.bayesian_estimates.size() > 0) {
        ImGui::Text("Latest Position Estimates:");
        ImGui::Separator();
        
        double bay = state.bayesian_estimates.back();
        double geo = state.geometric_estimates.back();
        double stat = state.statistical_estimates.back();
        double obs = state.observer_estimates.back();
        
        ImGui::TextColored(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), "Bayesian:        %.10f", bay);
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Geometric:       %.10f", geo);
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.8f, 1.0f), "Statistical:     %.10f", stat);
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Optimal Observer:%.10f", obs);
        
        ImGui::Separator();
        
        double max_diff = std::max({std::abs(bay - geo), std::abs(bay - stat), std::abs(bay - obs)});
        ImGui::Text("Maximum Difference: %.2e", max_diff);
        
        if (max_diff < 1e-10) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 
                "All implementations are NUMERICALLY IDENTICAL!");
        }
        
        ImGui::Spacing();
        ImGui::TextWrapped(
            "This proves that whether you derive the Kalman Filter from "
            "probability theory, linear algebra, optimization, or control theory, "
            "you get the SAME mathematical result!");
        
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
            "Run simulation to compare implementations");
    }
    
    ImGui::End();
}

/**
 * @brief Main function - initializes GLFW/ImGui and runs the main loop
 */
int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        return -1;
    }
    
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1400, 900, "Kalman Filter Foundations - Educational GUI", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Our state
    SimulationState state;
    float lastSimTime = 0.0f;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Auto-stepping logic
        if (state.running && !state.paused && state.current_step < state.max_steps) {
            float currentTime = static_cast<float>(glfwGetTime());
            float interval = 1.0f / state.simulation_speed;
            if (currentTime - lastSimTime >= interval) {
                state.step();
                lastSimTime = currentTime;
            }
        }
        
        // Render our windows
        renderMainWindow(state);
        renderPlotWindow(state);
        renderVelocityPlot(state);
        renderEducationalWindow();
        renderComparisonWindow(state);
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
