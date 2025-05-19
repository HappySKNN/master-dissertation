#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <string>
#include <vector>

#include "../External/matplotlibcpp.h"

namespace plt = matplotlibcpp;

/**
 * @brief Plots time series data with true values and smoothed/forecasted values
 * @param y_true Vector of true time series values
 * @param y_smooth Vector of smoothed or forecasted values 
 * @param title Title of the plot
 */
inline void plot_series(const std::vector<double> &y_true,
                        const std::vector<double> &y_smooth,
                        const std::string &title) {
    std::vector<double> t(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        t[i] = i + 1;
    }

    plt::figure_size(1200, 500);
    plt::named_plot("True", t, y_true, "b-");
    plt::named_plot("BSTS forecast", t, y_smooth, "r-");
    plt::title(title);
    plt::legend();
    plt::xlabel("Time");
    plt::ylabel("Value");
    plt::grid(true);
    plt::show();
}

#endif
