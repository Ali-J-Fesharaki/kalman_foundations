/**
 * @file main.cpp
 * @brief Pure Qt GUI for demonstrating Kalman Filter derivations
 *
 * This program implements a 1D constant velocity tracking problem and runs it
 * through all four mathematical formulations of the Kalman Filter, displaying
 * results using a pure Qt GUI with charts visualization.
 *
 * 1. Bayesian (Probability Density Functions)
 * 2. Geometric (Orthogonal Projection)
 * 3. Statistical (Minimum Mean Squared Error)
 * 4. Optimal Observer (Control Theory)
 *
 * The key insight is that despite their different mathematical origins,
 * all four approaches produce EXACTLY the same numerical results.
 */

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QGroupBox>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QLegend>

#include <cmath>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>

// Include all four implementations
#include "bayesian/KalmanFilter.hpp"
#include "geometric/KalmanFilter.hpp"
#include "statistical/KalmanFilter.hpp"
#include "optimal_observer/KalmanFilter.hpp"

/**
 * @brief Checks if two vectors are numerically equal within tolerance
 */
bool vectorsEqual(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tolerance = 1e-12)
{
    if (a.size() != b.size()) return false;
    return (a - b).norm() < tolerance;
}

/**
 * @brief Checks if two matrices are numerically equal within tolerance
 */
bool matricesEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tolerance = 1e-12)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    return (a - b).norm() < tolerance;
}

// Random seed for reproducible simulation results
constexpr unsigned int RANDOM_SEED = 42;

/**
 * @class KalmanFilterDemo
 * @brief Main window for the Kalman Filter demonstration using pure Qt
 */
class KalmanFilterDemo : public QMainWindow
{
    Q_OBJECT

public:
    KalmanFilterDemo(QWidget *parent = nullptr) : QMainWindow(parent)
    {
        setWindowTitle("Kalman Filter: Four Mathematical Derivations Demo (Pure Qt)");
        setMinimumSize(1200, 800);

        setupUI();
        runSimulation();
    }

private:
    QTextEdit *logOutput;
    QChartView *chartView;
    QChart *chart;
    QLineSeries *groundTruthSeries;
    QScatterSeries *measurementSeries;
    QLineSeries *bayesianSeries;
    QLineSeries *geometricSeries;
    QLineSeries *statisticalSeries;
    QLineSeries *observerSeries;
    QLabel *statusLabel;

    void setupUI()
    {
        QWidget *centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

        // Title
        QLabel *titleLabel = new QLabel("Kalman Filter: Four Mathematical Derivations");
        titleLabel->setAlignment(Qt::AlignCenter);
        QFont titleFont = titleLabel->font();
        titleFont.setPointSize(16);
        titleFont.setBold(true);
        titleLabel->setFont(titleFont);
        mainLayout->addWidget(titleLabel);

        // Subtitle
        QLabel *subtitleLabel = new QLabel("Demonstrating equivalence of Bayesian, Geometric, Statistical, and Control Theory approaches");
        subtitleLabel->setAlignment(Qt::AlignCenter);
        mainLayout->addWidget(subtitleLabel);

        // Main content area with chart and log
        QHBoxLayout *contentLayout = new QHBoxLayout();

        // Chart area
        setupChart();
        chartView = new QChartView(chart);
        chartView->setRenderHint(QPainter::Antialiasing);
        chartView->setMinimumWidth(700);
        contentLayout->addWidget(chartView, 2);

        // Log output area
        QVBoxLayout *logLayout = new QVBoxLayout();
        QLabel *logLabel = new QLabel("Simulation Log:");
        logLabel->setFont(QFont(logLabel->font().family(), 10, QFont::Bold));
        logLayout->addWidget(logLabel);

        logOutput = new QTextEdit();
        logOutput->setReadOnly(true);
        logOutput->setFont(QFont("Monospace", 9));
        logOutput->setMinimumWidth(400);
        logLayout->addWidget(logOutput);

        contentLayout->addLayout(logLayout, 1);
        mainLayout->addLayout(contentLayout);

        // Status bar
        statusLabel = new QLabel();
        statusLabel->setAlignment(Qt::AlignCenter);
        mainLayout->addWidget(statusLabel);

        // Button area
        QHBoxLayout *buttonLayout = new QHBoxLayout();
        buttonLayout->addStretch();

        QPushButton *runButton = new QPushButton("Run Simulation");
        runButton->setMinimumWidth(150);
        connect(runButton, &QPushButton::clicked, this, &KalmanFilterDemo::runSimulation);
        buttonLayout->addWidget(runButton);

        QPushButton *clearButton = new QPushButton("Clear");
        clearButton->setMinimumWidth(100);
        connect(clearButton, &QPushButton::clicked, this, &KalmanFilterDemo::clearResults);
        buttonLayout->addWidget(clearButton);

        buttonLayout->addStretch();
        mainLayout->addLayout(buttonLayout);
    }

    void setupChart()
    {
        chart = new QChart();
        chart->setTitle("1D Constant Velocity Tracking - Position over Time");

        // Ground truth series
        groundTruthSeries = new QLineSeries();
        groundTruthSeries->setName("Ground Truth");
        groundTruthSeries->setPen(QPen(Qt::black, 2, Qt::DashLine));
        chart->addSeries(groundTruthSeries);

        // Measurement series (scatter points)
        measurementSeries = new QScatterSeries();
        measurementSeries->setName("Measurements");
        measurementSeries->setMarkerSize(8);
        measurementSeries->setColor(Qt::red);
        chart->addSeries(measurementSeries);

        // Filter estimate series
        bayesianSeries = new QLineSeries();
        bayesianSeries->setName("Bayesian");
        bayesianSeries->setPen(QPen(Qt::blue, 2));
        chart->addSeries(bayesianSeries);

        geometricSeries = new QLineSeries();
        geometricSeries->setName("Geometric");
        geometricSeries->setPen(QPen(Qt::green, 2));
        chart->addSeries(geometricSeries);

        statisticalSeries = new QLineSeries();
        statisticalSeries->setName("Statistical");
        statisticalSeries->setPen(QPen(QColor(255, 165, 0), 2));  // Orange
        chart->addSeries(statisticalSeries);

        observerSeries = new QLineSeries();
        observerSeries->setName("Optimal Observer");
        observerSeries->setPen(QPen(Qt::magenta, 2));
        chart->addSeries(observerSeries);

        // Set up axes
        QValueAxis *axisX = new QValueAxis();
        axisX->setTitleText("Time Step");
        axisX->setRange(-0.5, 10.5);
        axisX->setTickCount(12);
        chart->addAxis(axisX, Qt::AlignBottom);

        QValueAxis *axisY = new QValueAxis();
        axisY->setTitleText("Position");
        axisY->setRange(-2, 12);
        chart->addAxis(axisY, Qt::AlignLeft);

        // Attach all series to axes
        groundTruthSeries->attachAxis(axisX);
        groundTruthSeries->attachAxis(axisY);
        measurementSeries->attachAxis(axisX);
        measurementSeries->attachAxis(axisY);
        bayesianSeries->attachAxis(axisX);
        bayesianSeries->attachAxis(axisY);
        geometricSeries->attachAxis(axisX);
        geometricSeries->attachAxis(axisY);
        statisticalSeries->attachAxis(axisX);
        statisticalSeries->attachAxis(axisY);
        observerSeries->attachAxis(axisX);
        observerSeries->attachAxis(axisY);

        chart->legend()->setVisible(true);
        chart->legend()->setAlignment(Qt::AlignBottom);
    }

    void log(const QString& message)
    {
        logOutput->append(message);
    }

private slots:
    void runSimulation()
    {
        clearResults();

        log("=========================================================");
        log("   Kalman Filter: Four Mathematical Derivations Demo");
        log("=========================================================\n");

        // System Definition
        const int state_dim = 2;
        const int meas_dim = 1;
        double dt = 1.0;

        Eigen::MatrixXd F(state_dim, state_dim);
        F << 1, dt,
             0, 1;

        Eigen::MatrixXd H(meas_dim, state_dim);
        H << 1, 0;

        double q = 0.1;
        Eigen::MatrixXd Q(state_dim, state_dim);
        Q << (q * dt * dt * dt / 3), (q * dt * dt / 2),
             (q * dt * dt / 2),      (q * dt);

        double r = 1.0;
        Eigen::MatrixXd R(meas_dim, meas_dim);
        R << r;

        Eigen::VectorXd x0(state_dim);
        x0 << 0.0, 1.0;

        Eigen::MatrixXd P0(state_dim, state_dim);
        P0 << 1.0, 0.0,
              0.0, 1.0;

        // Generate measurements
        Eigen::VectorXd x_true(state_dim);
        x_true << 0.0, 1.0;

        std::mt19937 rng(RANDOM_SEED);
        std::normal_distribution<double> meas_noise(0.0, std::sqrt(r));

        const int num_steps = 10;
        std::vector<double> measurements(num_steps);

        log("Ground Truth & Measurements:");
        log("----------------------------");
        log(QString("%1 %2 %3").arg("Step", 6).arg("True Pos", 12).arg("Measured", 12));

        Eigen::VectorXd x_current = x_true;
        for (int k = 0; k < num_steps; ++k)
        {
            if (k > 0)
            {
                x_current = F * x_current;
            }

            double true_pos = x_current(0);
            double meas = true_pos + meas_noise(rng);
            measurements[k] = meas;

            // Add to chart
            groundTruthSeries->append(k, true_pos);
            measurementSeries->append(k, meas);

            log(QString("%1 %2 %3")
                .arg(k, 6)
                .arg(true_pos, 12, 'f', 4)
                .arg(meas, 12, 'f', 4));
        }
        log("");

        // Initialize filters
        bayesian::KalmanFilter kf_bayesian(state_dim, meas_dim);
        geometric::KalmanFilter kf_geometric(state_dim, meas_dim);
        statistical::KalmanFilter kf_statistical(state_dim, meas_dim);
        optimal_observer::KalmanFilter kf_observer(state_dim, meas_dim);

        auto configureFilter = [&](auto& kf) {
            kf.setStateTransition(F);
            kf.setMeasurementMatrix(H);
            kf.setProcessNoise(Q);
            kf.setMeasurementNoise(R);
            kf.setState(x0);
            kf.setCovariance(P0);
        };

        configureFilter(kf_bayesian);
        configureFilter(kf_geometric);
        configureFilter(kf_statistical);
        configureFilter(kf_observer);

        log("Running Kalman Filters...");
        log("=========================\n");

        bool all_match = true;

        for (int k = 0; k < num_steps; ++k)
        {
            log(QString("--- Step %1 ---").arg(k));

            Eigen::VectorXd z(meas_dim);
            z << measurements[k];

            kf_bayesian.predict();
            kf_geometric.predict();
            kf_statistical.predict();
            kf_observer.predict();

            kf_bayesian.update(z);
            kf_geometric.update(z);
            kf_statistical.update(z);
            kf_observer.update(z);

            const auto& x_bay = kf_bayesian.getState();
            const auto& x_geo = kf_geometric.getState();
            const auto& x_stat = kf_statistical.getState();
            const auto& x_obs = kf_observer.getState();

            // Add to chart
            bayesianSeries->append(k, x_bay(0));
            geometricSeries->append(k, x_geo(0));
            statisticalSeries->append(k, x_stat(0));
            observerSeries->append(k, x_obs(0));

            const auto& P_bay = kf_bayesian.getCovariance();
            const auto& P_geo = kf_geometric.getCovariance();
            const auto& P_stat = kf_statistical.getCovariance();
            const auto& P_obs = kf_observer.getCovariance();

            log(QString("       Bayesian: [%1, %2]").arg(x_bay(0), 0, 'f', 6).arg(x_bay(1), 0, 'f', 6));
            log(QString("      Geometric: [%1, %2]").arg(x_geo(0), 0, 'f', 6).arg(x_geo(1), 0, 'f', 6));
            log(QString("    Statistical: [%1, %2]").arg(x_stat(0), 0, 'f', 6).arg(x_stat(1), 0, 'f', 6));
            log(QString("Optimal Observer: [%1, %2]").arg(x_obs(0), 0, 'f', 6).arg(x_obs(1), 0, 'f', 6));

            bool states_match =
                vectorsEqual(x_bay, x_geo) &&
                vectorsEqual(x_bay, x_stat) &&
                vectorsEqual(x_bay, x_obs);

            bool covs_match =
                matricesEqual(P_bay, P_geo) &&
                matricesEqual(P_bay, P_stat) &&
                matricesEqual(P_bay, P_obs);

            if (states_match && covs_match)
            {
                log("✓ All four implementations produce IDENTICAL results!\n");
            }
            else
            {
                log("✗ MISMATCH DETECTED!\n");
                all_match = false;
            }
        }

        // Final summary
        log("=========================================================");
        log("                     FINAL SUMMARY");
        log("=========================================================\n");

        log("Final state estimates:");
        log(QString("       Bayesian: [%1, %2]").arg(kf_bayesian.getState()(0), 0, 'f', 6).arg(kf_bayesian.getState()(1), 0, 'f', 6));
        log(QString("      Geometric: [%1, %2]").arg(kf_geometric.getState()(0), 0, 'f', 6).arg(kf_geometric.getState()(1), 0, 'f', 6));
        log(QString("    Statistical: [%1, %2]").arg(kf_statistical.getState()(0), 0, 'f', 6).arg(kf_statistical.getState()(1), 0, 'f', 6));
        log(QString("Optimal Observer: [%1, %2]").arg(kf_observer.getState()(0), 0, 'f', 6).arg(kf_observer.getState()(1), 0, 'f', 6));

        if (all_match)
        {
            log("\n=========================================================");
            log("  SUCCESS: All four derivations produce identical results!");
            log("=========================================================\n");

            log("This demonstrates that whether you view the Kalman Filter as:");
            log("  1. Bayesian inference (product of Gaussians)");
            log("  2. Orthogonal projection in Hilbert space");
            log("  3. Minimum Mean Squared Error estimation");
            log("  4. Optimal state observer design");
            log("\nThe mathematical formulas are equivalent!");

            statusLabel->setText("<b style='color: green;'>✓ SUCCESS: All four derivations produce identical results!</b>");
            statusLabel->setStyleSheet("color: green; font-size: 14px;");
        }
        else
        {
            log("\n=========================================================");
            log("  FAILURE: Results don't match - check implementation!");
            log("=========================================================");

            statusLabel->setText("<b style='color: red;'>✗ FAILURE: Results don't match - check implementation!</b>");
            statusLabel->setStyleSheet("color: red; font-size: 14px;");
        }
    }

    void clearResults()
    {
        logOutput->clear();
        groundTruthSeries->clear();
        measurementSeries->clear();
        bayesianSeries->clear();
        geometricSeries->clear();
        statisticalSeries->clear();
        observerSeries->clear();
        statusLabel->clear();
    }
};

// Need to include the moc file for Q_OBJECT
#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    KalmanFilterDemo window;
    window.show();

    return app.exec();
}
