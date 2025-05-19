#ifndef MCMC_H
#define MCMC_H

#include <vector>

struct MCMCResult {
    std::vector<std::vector<std::vector<double> > > alpha_samples; // [iteration][time][state]
    std::vector<double> phi_samples;
    std::vector<double> level_var_samples;
    std::vector<double> trend_var_samples;
    std::vector<double> season_var_samples;
    std::vector<double> obs_var_samples;
};

MCMCResult run_mcmc(const std::vector<double> &y,
                    const std::vector<double> &x,
                    int season_period,
                    int n_iter,
                    double prior_a,
                    double prior_b);

#endif
