#ifndef PRIOR_H
#define PRIOR_H

#include <vector>

/**
 * @class Prior
 * @brief Represents a prior distribution for the AR(1) model coefficient.
 */
class Prior {
    double mean; ///< Mean of the prior distribution
    double variance; ///< Variance of the prior distribution

    /**
     * @brief Estimates the AR(1) coefficient using Ordinary Least Squares (OLS).
     * @param series Time series data.
     * @return Estimated AR(1) coefficient.
     * @throws std::invalid_argument if there's insufficient data.
     */
    static double estimateOLS(const std::vector<double> &series);

public:
    /**
     * @brief Default constructor. Initializes a weakly informative prior N(0, 1).
     */
    Prior();

    /**
     * @brief Constructor for training data to estimate the prior (empirical Bayes).
     * @param train_data Training time series data (not used for forecasting).
     */
    explicit Prior(const std::vector<double> &train_data);

    /**
     * @brief Constructor for multiple training datasets (hierarchical Bayes).
     * @param train_datasets Multiple training time series datasets.
     * @throws std::invalid_argument if datasets are empty or invalid.
     */
    explicit Prior(const std::vector<std::vector<double> > &train_datasets);

    /**
     * @brief Gets the mean of the prior distribution.
     * @return Mean value.
     */
    [[nodiscard]] double getMean() const;

    /**
     * @brief Gets the variance of the prior distribution.
     * @return Variance value.
     */
    [[nodiscard]] double getVariance() const;
};

#endif
