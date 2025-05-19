#include "Utility/CSVLoader.h"
#include "MCMC.h"
#include "Utility/Graphics.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <cmath>

int main() {
    // === Загрузка и прореживание ===
    SeriesData data_src = load_series_csv("../Datasets/btc.csv");
    SeriesData data;
    for (int i = 0; i < data_src.y.size(); i += 50) {
        data.x.push_back(data_src.x[i]);
        data.y.push_back(data_src.y[i]);
    }

    int N = data.y.size();
    if (N < 20) {
        std::cerr << "Недостаточно данных!" << std::endl;
        return 1;
    }

    // === Разделение: 80% train, 20% test ===
    int split_idx = static_cast<int>(0.8 * N);
    std::vector<double> x_train(data.x.begin(), data.x.begin() + split_idx);
    std::vector<double> y_train(data.y.begin(), data.y.begin() + split_idx);
    std::vector<double> x_test(data.x.begin() + split_idx, data.x.end());
    std::vector<double> y_test(data.y.begin() + split_idx, data.y.end());

    std::cout << "Train: " << y_train.size() << " points, Test: " << y_test.size() << " points.\n";

    // === Настройки модели ===
    int season_period = 12;
    int n_iter = 500;
    double prior_a = 2.0, prior_b = 1.0;
    int burn = n_iter - 200;
    int forecast_horizon = static_cast<int>(y_test.size());

    // === Запуск обучения BSTS (MCMC) ===
    auto t1 = std::chrono::high_resolution_clock::now();
    MCMCResult mcmc = run_mcmc(y_train, x_train, season_period, n_iter, prior_a, prior_b);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    std::cout << "MCMC finished in " << dt.count() << " seconds\n";

    // === Прогноз на тестовые точки ===
    std::vector<double> forecasts;
    std::vector<double> test_x = x_test;

    for (int h = 0; h < forecast_horizon; ++h) {
        double pred_sum = 0.0;

        for (int i = burn; i < n_iter; ++i) {
            const auto& alpha = mcmc.alpha_samples[i].back();  // последнее скрытое состояние
            double level = alpha[0];
            double trend = alpha[1];
            double season = alpha[2];
            double phi = mcmc.phi_samples[i];

            double pred = level + trend + season + phi * test_x[h];
            pred_sum += pred;
        }

        forecasts.push_back(pred_sum / (n_iter - burn));
    }

    // === Расчёт MAE ===
    double mae = 0.0;
    for (size_t i = 0; i < y_test.size(); ++i)
        mae += std::abs(y_test[i] - forecasts[i]);
    mae /= static_cast<double>(y_test.size());

    std::cout << "MAE on test set: " << mae << std::endl;

    // === Подготовка данных для графика ===
    const std::vector<double>& full_y = y_test;
    std::vector<double> full_pred(y_test.size(), NAN);
    for (size_t i = 0; i < forecasts.size(); ++i)
        full_pred[i] = forecasts[i];

    // === График ===
    plot_series(full_y, full_pred, "BSTS Forecast vs Actual (test period)");

    return 0;
}
