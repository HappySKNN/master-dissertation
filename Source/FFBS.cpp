#include "../Include/FFBS.h"
#include <random>
#include <Eigen/Dense>

typedef std::vector<double> vec;
typedef std::vector<vec> mat;

static vec zeros(const int d) { return vec(d, 0.0); }

static mat eye(const int d) {
    mat M(d, vec(d, 0.0));
    for (int i = 0; i < d; ++i) M[i][i] = 1.0;
    return M;
}

static vec matvec(const mat &M, const vec &v) {
    const int d = v.size();
    const int k = M.size();
    vec r(k, 0.0);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < d; ++j)
            r[i] += M[i][j] * v[j];
    return r;
}

static mat matmat(const mat &A, const mat &B) {
    int n = A.size(), m = B[0].size(), l = B.size();
    mat res(n, vec(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < l; ++k)
                res[i][j] += A[i][k] * B[k][j];
    return res;
}

static mat transp(const mat &A) {
    int n = A.size(), m = A[0].size();
    mat res(m, vec(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            res[j][i] = A[i][j];
    return res;
}

static vec add(const vec &a, const vec &b) {
    int d = a.size();
    vec r(d, 0.0);
    for (int i = 0; i < d; ++i) r[i] = a[i] + b[i];
    return r;
}

static vec scal(const vec &a, double s) {
    int d = a.size();
    vec r(d, 0.0);
    for (int i = 0; i < d; ++i) r[i] = a[i] * s;
    return r;
}

static mat diag(const vec &q) {
    int d = q.size();
    mat Q(d, vec(d, 0.0));
    for (int i = 0; i < d; ++i) Q[i][i] = q[i];
    return Q;
}

static double sample_normal(double m, double s, std::mt19937 &gen) {
    std::normal_distribution<> nd(m, s);
    return nd(gen);
}

static vec sample_mvnorm(const vec &mu, const mat &Sigma, std::mt19937 &gen) {
    int d = mu.size();
    Eigen::MatrixXd S(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            S(i, j) = Sigma[i][j];
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    Eigen::VectorXd z = Eigen::VectorXd::NullaryExpr(d, [&]() { return sample_normal(0.0, 1.0, gen); });
    Eigen::VectorXd x = llt.matrixL() * z;
    vec out(d, 0.0);
    for (int i = 0; i < d; ++i)
        out[i] = mu[i] + x[i];
    return out;
}

FFBSResult ffbs_bsts(const std::vector<double> &y,
                     const std::vector<double> &x,
                     double phi,
                     int season_period,
                     double level_var,
                     double trend_var,
                     double season_var,
                     double obs_var) {
    int N = y.size();
    int d = 2 + (season_period - 1);

    mat T(d, vec(d, 0.0));
    T[0][0] = 1.0;
    T[0][1] = 1.0;
    T[1][1] = 1.0;
    for (int i = 2; i < d - 1; ++i)
        T[i][i + 1] = 1.0;
    for (int i = 2; i < d; ++i)
        T[d - 1][i] = -1.0;

    vec q(d, 0.0);
    q[0] = level_var;
    q[1] = trend_var;
    q[d - 1] = season_var;
    mat Q = diag(q);

    vec Z(d, 0.0);
    Z[0] = 1.0;
    Z[1] = 1.0;
    Z[2] = 1.0;

    std::vector<vec> m(N), a(N);
    std::vector<mat> C(N), R(N);
    m[0] = zeros(d);
    C[0] = eye(d);

    for (int t = 1; t < N; ++t) {
        a[t] = matvec(T, m[t - 1]);
        mat TC = matmat(T, C[t - 1]);
        mat R_ = matmat(TC, transp(T));
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                R_[i][j] += Q[i][j];
        R[t] = R_;

        double pred = 0.0;
        for (int i = 0; i < d; ++i) pred += Z[i] * a[t][i];
        pred += phi * x[t];
        double v = y[t] - pred;

        double f = 0.0;
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                f += Z[i] * R[t][i][j] * Z[j];
        f += obs_var;

        vec K(d, 0.0);
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j)
                K[i] += R[t][i][j] * Z[j];
            K[i] /= f;
        }
        m[t] = add(a[t], scal(K, v));
        C[t] = R[t];
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[t][i][j] -= K[i] * K[j] * f;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<vec> alpha(N, zeros(d));
    alpha[N - 1] = sample_mvnorm(m[N - 1], C[N - 1], gen);
    for (int t = N - 2; t >= 0; --t) {
        Eigen::MatrixXd Ct(d, d), Tt(d, d), Rt(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                Ct(i, j) = C[t][i][j];
                Tt(i, j) = T[j][i];
                Rt(i, j) = R[t + 1][i][j];
            }
        Eigen::MatrixXd J = Ct * Tt * Rt.inverse();
        Eigen::VectorXd atp1(d), alpha_tp1(d), mt(d);
        for (int i = 0; i < d; ++i) {
            atp1(i) = a[t + 1][i];
            alpha_tp1(i) = alpha[t + 1][i];
            mt(i) = m[t][i];
        }
        Eigen::VectorXd mean = mt + J * (alpha_tp1 - atp1);
        Eigen::MatrixXd cov = Ct - J * Rt * J.transpose();
        Eigen::LLT<Eigen::MatrixXd> llt(cov);
        Eigen::VectorXd z = Eigen::VectorXd::NullaryExpr(d, [&]() { return sample_normal(0.0, 1.0, gen); });
        Eigen::VectorXd x_samp = llt.matrixL() * z;
        for (int i = 0; i < d; ++i)
            alpha[t][i] = mean(i) + x_samp(i);
    }

    FFBSResult res;
    res.filtered_means = m;
    res.filtered_covs = C;
    res.sampled_states = alpha;
    return res;
}
