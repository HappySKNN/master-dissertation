#include "../Include/Prior.h"
#include <iostream>
#include <stdexcept>

Prior::Prior() : mean(0.0), variance(1.0) {
}

Prior::Prior(const std::vector<double> &train_data) {
    // Проверка на достаточность данных для оценки дисперсии
    const size_t n = train_data.size() - 1;
    if (n < 1) {
        throw std::invalid_argument("Insufficient data for AR(1) estimation");
    }

    // Оценка α
    const double alpha_hat = estimateOLS(train_data);

    // Проверка на возможность расчета несмещенной оценки σ²
    if (n == 1) {
        throw std::invalid_argument("At least 3 data points required for variance estimation");
    }

    // Оценка σ² через остатки
    double sigma_sq = 0.0;
    for (size_t t = 1; t < train_data.size(); ++t) {
        const double residual = train_data[t] - alpha_hat * train_data[t - 1];
        sigma_sq += residual * residual;
    }
    sigma_sq /= static_cast<double>(n - 1); // Несмещенная оценка

    // Проверка на нулевую сумму квадратов лагов
    double sum_y_lag_sq = 0.0;
    for (size_t t = 1; t < train_data.size(); ++t) {
        sum_y_lag_sq += train_data[t - 1] * train_data[t - 1];
    }
    if (sum_y_lag_sq == 0.0) {
        throw std::invalid_argument("Sum of squared lags is zero");
    }

    // Расчет дисперсии априора
    const double var_alpha = sigma_sq / sum_y_lag_sq;

    this->mean = alpha_hat;
    this->variance = var_alpha;
}

Prior::Prior(const std::vector<std::vector<double> > &train_datasets) {
    if (train_datasets.empty()) {
        throw std::invalid_argument("No datasets provided");
    }

    std::vector<double> alpha_estimates;
    for (const auto &series: train_datasets) {
        try {
            double alpha = estimateOLS(series);
            alpha_estimates.push_back(alpha);
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error processing series: " << e.what() << std::endl;
        }
    }

    if (alpha_estimates.empty()) {
        throw std::invalid_argument("All datasets are invalid");
    }

    // Расчет среднего и дисперсии
    double sum = 0.0;
    for (const double alpha: alpha_estimates) {
        sum += alpha;
    }
    mean = sum / static_cast<double>(alpha_estimates.size());

    double var = 0.0;
    for (const double alpha: alpha_estimates) {
        var += (alpha - mean) * (alpha - mean);
    }
    variance = var / static_cast<double>(alpha_estimates.size());
}

double Prior::estimateOLS(const std::vector<double> &series) {
    if (const size_t n = series.size() - 1; n < 1) {
        throw std::invalid_argument("Insufficient data for AR(1) estimation");
    }

    double sum_lag_current = 0.0; // sum(y_{t-1} * y_t)
    double sum_lag_squared = 0.0; // sum(y_{t-1}^2)

    for (size_t t = 1; t < series.size(); ++t) {
        const double y_lag = series[t - 1];
        sum_lag_current += y_lag * series[t];
        sum_lag_squared += y_lag * y_lag;
    }

    return sum_lag_current / sum_lag_squared;
}

double Prior::getMean() const {
    return mean;
}

double Prior::getVariance() const {
    return variance;
}
