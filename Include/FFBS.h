#ifndef FFBS_H
#define FFBS_H

#include <vector>

struct FFBSResult {
    std::vector<std::vector<double> > filtered_means;
    std::vector<std::vector<std::vector<double> > > filtered_covs;
    std::vector<std::vector<double> > sampled_states;
};

FFBSResult ffbs_bsts(const std::vector<double> &y,
                     const std::vector<double> &x,
                     double phi,
                     int season_period,
                     double level_var,
                     double trend_var,
                     double season_var,
                     double obs_var);

#endif
