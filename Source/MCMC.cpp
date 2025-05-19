#include "../Include/MCMC.h"
#include "FFBS.h"
#include <random>
#include <cmath>
#include <iostream>

static double sample_inv_gamma(double a, double b, std::mt19937 &gen) {
    std::gamma_distribution<> gdist(a, 1.0 / b);
    return 1.0 / gdist(gen);
}

MCMCResult run_mcmc(const std::vector<double> &y,
                    const std::vector<double> &x,
                    int season_period,
                    int n_iter,
                    double prior_a,
                    double prior_b) {
    int N = y.size();
    int d = 2 + (season_period - 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Инициализация параметров
    double level_var = 0.1, trend_var = 0.1, season_var = 0.1, obs_var = 0.1;
    double phi = 1.0;

    MCMCResult res;

    for (int iter = 0; iter < n_iter; ++iter) {
        // --- 1. FFBS: сэмплируем скрытые состояния (на текущих параметрах)
        FFBSResult ffbs = ffbs_bsts(y, x, phi, season_period, level_var, trend_var, season_var, obs_var);
        const auto &alpha = ffbs.sampled_states;

        // --- 2. Обновляем дисперсии
        double sum_level = 0, sum_trend = 0, sum_season = 0, sum_obs = 0;
        for (int t = 1; t < N; ++t)
            sum_level += pow(alpha[t][0] - alpha[t - 1][0] - alpha[t - 1][1], 2);
        for (int t = 1; t < N; ++t)
            sum_trend += pow(alpha[t][1] - alpha[t - 1][1], 2);
        for (int t = season_period; t < N; ++t) {
            double sum_prev = 0.0;
            for (int j = 1; j < season_period; ++j)
                sum_prev += alpha[t - j][2];
            sum_season += pow(alpha[t][2] + sum_prev, 2);
        }
        for (int t = 0; t < N; ++t) {
            double pred = alpha[t][0] + alpha[t][1] + alpha[t][2] + phi * x[t];
            sum_obs += pow(y[t] - pred, 2);
        }

        level_var = sample_inv_gamma(prior_a + 0.5 * (N - 1), prior_b + 0.5 * sum_level, gen);
        trend_var = sample_inv_gamma(prior_a + 0.5 * (N - 1), prior_b + 0.5 * sum_trend, gen);
        season_var = sample_inv_gamma(prior_a + 0.5 * (N - season_period), prior_b + 0.5 * sum_season, gen);
        obs_var = sample_inv_gamma(prior_a + 0.5 * N, prior_b + 0.5 * sum_obs, gen);

        // --- 3. Обновляем phi (коэфф. регрессора, условная нормаль)
        double xtx = 0.0, xty = 0.0;
        for (int t = 0; t < N; ++t) {
            double resid = y[t] - alpha[t][0] - alpha[t][1] - alpha[t][2];
            xtx += x[t] * x[t];
            xty += x[t] * resid;
        }
        double var_post = 1.0 / (xtx / obs_var + 1.0);
        double mean_post = var_post * (xty / obs_var);
        std::normal_distribution<> ndist(mean_post, std::sqrt(var_post));
        phi = ndist(gen);

        // --- 4. Сохраняем сэмплы
        res.alpha_samples.push_back(alpha);
        res.level_var_samples.push_back(level_var);
        res.trend_var_samples.push_back(trend_var);
        res.season_var_samples.push_back(season_var);
        res.obs_var_samples.push_back(obs_var);
        res.phi_samples.push_back(phi);

        if (iter % 100 == 0)
            std::cout << "[MCMC] Iter " << iter << " level_var: " << level_var
                    << " trend_var: " << trend_var << " season_var: " << season_var
                    << " obs_var: " << obs_var << " phi: " << phi << std::endl;
    }

    return res;
}
