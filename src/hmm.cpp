#include "hmm.h"
#include <iomanip>

namespace hmm {

static constexpr auto NaN = std::numeric_limits<HMM_DATA_T>::quiet_NaN();
static constexpr auto inf = std::numeric_limits<HMM_DATA_T>::infinity();

/*
 * Compute the sum of two log-probabilities. See [4] for more information.
 */
template<std::floating_point T>
auto log_add(T log_x, T log_y)
{
    if (log_x == -inf)
        return log_y;
    if (log_y == -inf)
        return log_x;

    return log_x + std::log(T {1} + std::exp(log_y-log_x));
}

template<class T>
auto batched_log(T &xs)
{
    for (auto &x: xs)
        x = std::log(x);
}

template<class T>
auto batched_exp(T &xs)
{
    for (auto &x: xs)
        x = std::exp(x);
}

auto normalize_chunks(HMM_SIZE_T N, HMM_SIZE_T M, HMM_DATA_T *p)
{
    for (HMM_SIZE_T i {}; i < N; ++i) {
        const auto base = i * M;
        auto accum = -inf;
        for (HMM_SIZE_T j {}; j < M; ++j)
            accum = log_add(accum, p[base + j]);
        for (HMM_SIZE_T j {}; j < M; ++j)
            p[base + j] -= accum;
    }
}

[[nodiscard]]
auto allocate_xi(HMM_SIZE_T N, HMM_SIZE_T T)
{
    return std::vector<matrix<HMM_DATA_T>> {T - 1, data_matrix_t {N, N}};
}

/*
 * Determine the most-likely state sequence given a sequence of observations.
 */
[[nodiscard]]
auto viterbi(const size_vector_t &X, const model_parameters &model)
{
    const auto &[A, B, pi] = model;
    const auto N = B.nrows();
    const auto T = X.size();
    data_matrix_t delta {T, N};
    matrix<HMM_SIZE_T> psi {T, N};

    for (HMM_SIZE_T i {}; i < N; ++i)
        delta(0, i) = pi(i) + B(i, X(0));

    for (HMM_SIZE_T t {1}; t < T; ++t) {
        for (HMM_SIZE_T j {}; j < N; ++j) {
            const auto b = B(j, X(t));
            auto value = -inf;
            HMM_SIZE_T index {};

            for (HMM_SIZE_T i {}; i < N; ++i) {
                if (const auto p = delta(t - 1, i) + A(i, j) + b; p > value) {
                    value = p;
                    index = i;
                }
            }
            delta(t, j) = value;
            psi(t, j) = index;
        }
    }

    auto value = -inf;
    HMM_SIZE_T index {};

    // Find the end of the most-probable state path.
    for (HMM_SIZE_T i {}; i < N; ++i) {
        if (delta(T - 1, i) > value) {
            value = delta(T - 1, i);
            index = i;
        }
    }

    // Backtrack through the pointers array.
    size_vector_t path {T};
    for (HMM_SIZE_T i {}; i < T; ++i) {
        const auto t = T - i - 1;
        path(t) = index;
        index = psi(t, index);
    }
    return predict_result {delta, path};
}

/*
 * Compute the forward parameter, alpha.
 */
[[nodiscard]]
auto forward(const size_vector_t &X, const model_parameters &model) -> matrix<HMM_DATA_T>
{
    const auto &[A, B, pi] = model;
    const auto N = A.nrows();
    const auto T = X.size();
    data_matrix_t alpha {T, N};

    for (HMM_SIZE_T i {}; i < N; ++i)
        alpha(0, i) = pi(i) + B(i, X(0));

    for (HMM_SIZE_T t {1}; t < T; ++t) {
        for (HMM_SIZE_T j {}; j < N; ++j) {
            auto accum = -inf;
            for (HMM_SIZE_T i {}; i < N; ++i)
                accum = log_add(accum, alpha(t - 1, i) + A(i, j));
            // Emission term can be factored out of the previous summation since it corresponds to
            // the destination state.
            alpha(t, j) = accum + B(j, X(t));
        }
    }
    return alpha;
}

/*
 * Compute the backward parameter, beta.
 */
[[nodiscard]]
auto backward(const size_vector_t &X, const model_parameters &model) -> matrix<HMM_DATA_T>
{
    const auto &[A, B, pi] = model;
    const auto N = A.nrows();
    const auto T = X.size();
    // Note that the last column is already initialized to log(1) = 0.
    data_matrix_t beta {T, N};

    for (HMM_SIZE_T n {}, t {T - 2}; n < T - 1; ++n, --t) {
        for (HMM_SIZE_T i {}; i < N; ++i) {
            auto accum = -inf;
            for (HMM_SIZE_T j {}; j < N; ++j)
                accum = log_add(accum, beta(t + 1, j) + A(i, j) + B(j, X(t + 1)));
            // Emission term cannot be factored out this time since the loop varies on the destination state,
            // which is where the symbol is emitted.
            beta(t, i) = accum;
        }
    }
    return beta;
}

/*
 * Compute the state posterior probability, gamma.
 */
[[nodiscard]]
auto state_posterior(const data_matrix_t &alpha, const data_matrix_t &beta)
{
    const auto T = alpha.nrows();
    const auto N = alpha.ncols();
    data_matrix_t gamma {T, N};
    
    for (HMM_SIZE_T t {}; t < T; ++t) {
        auto accum = -inf;
        for (HMM_SIZE_T i {}; i < N; ++i) {
            gamma(t, i) = alpha(t, i) + beta(t, i);
            accum = log_add(accum, gamma(t, i));
        }
        for (HMM_SIZE_T i {}; i < N; ++i)
            gamma(t, i) -= accum;
    }
    return gamma;
}

/*
 * Compute the segment posterior probability, xi.
 */
[[nodiscard]]
auto segment_posterior(const size_vector_t &X, const model_parameters &model, const data_matrix_t &alpha, const data_matrix_t &beta)
{
    const auto &[A, B, pi] = model;
    const auto T = alpha.nrows();
    const auto N = alpha.ncols();
    auto xi = allocate_xi(N, T);

    for (HMM_SIZE_T t {}; t < T - 1; ++t) {
        auto den = -inf;
        for (HMM_SIZE_T i {}; i < N; ++i) {
            for (HMM_SIZE_T j {}; j < N; ++j) {
                const auto num = alpha(t, i) + A(i, j) + B(j, X(t + 1)) + beta(t + 1, j);
                den = log_add(den, num);
                xi[t](i, j) = num;
            }
        }
        for (HMM_SIZE_T i {}; i < N; ++i) {
            for (HMM_SIZE_T j {}; j < N; ++j)
                xi[t](i, j) -= den;
        }
    }
    return xi;
}

[[nodiscard]]
auto update_model(const size_vector_t &X, model_parameters &model, const data_matrix_t &gamma, const std::vector<matrix<HMM_DATA_T>> &xi, const fix_parameters &fixed)
{
    auto &[A, B, pi] = model;
    const auto N = A.nrows();
    const auto M = B.ncols();
    const auto T = X.size();

    // Update the transition probabilities. Note that gamma is equivalent to xi marginalized
    // over the destination states.
    //
    //            # transitions from state i to state j
    // A(i, j) = ---------------------------------------
    //            # transitions from state i to state *
    //
    if (!fixed.values[fix_parameters::TRANSITION]) {
        for (HMM_SIZE_T i {}; i < N; ++i) {
            for (HMM_SIZE_T j {}; j < N; ++j) {
                auto num = -inf;
                auto den = -inf;
                for (HMM_SIZE_T t {}; t < T - 1; ++t) {
                    num = log_add(num, xi[t](i, j));
                    den = log_add(den, gamma(t, i));
                }
                A(i, j) = num - den;
            }
        }
    }

    // Update the emission probabilities.
    //
    //            # emissions of symbol k from state j
    // B(j, k) = --------------------------------------
    //            # emissions of symbol * from state j
    //
    if (!fixed.values[fix_parameters::EMISSION]) {
        for (HMM_SIZE_T j {}; j < N; ++j) {
            for (HMM_SIZE_T k {}; k < M; ++k) {
                auto num = -inf;
                auto den = -inf;
                for (HMM_SIZE_T t {}; t < T; ++t) {
                    num = X(t) == k
                        ? log_add(num, gamma(t, j)) : num;
                    den = log_add(den, gamma(t, j));
                }
                B(j, k) = num - den;
            }
        }
    }

    // Update the initial distribution.
    if (!fixed.values[fix_parameters::INITIAL]) {
        for (HMM_SIZE_T i {}; i < N; ++i)
            pi(i) = gamma(0, i);
    }
}

auto load_parameters(std::istream &is) -> model_parameters
{
    std::uint64_t N;
    std::uint64_t M;

    const auto try_read = [&is](auto &data, const auto &what) {
        if (!(is >> data))
            throw std::runtime_error {std::string{"unable to read "} + what};
    };

    try_read(N, "state count");
    try_read(M, "symbol count");

    data_matrix_t A {N, N};
    for (auto &value: A)
        try_read(value, "transition probability");

    data_matrix_t B {N, M};
    for (auto &value: B)
        try_read(value, "emission probability");

    data_vector_t pi {N};
    for (auto &value: pi)
        try_read(value, "initial distribution value");

    return model_parameters {A, B, pi};
}

auto save_parameters(std::ostream &os, const model_parameters &theta, bool readable) -> void
{
    static constexpr auto precision = std::numeric_limits<HMM_DATA_T>::max_digits10 + 2;
    os << std::setprecision(precision);

    const auto try_write = [&os](auto data, const auto &what) {
        if (!(os << data))
            throw std::runtime_error {std::string{"unable to write "} + what};
    };

    const auto add_hard_separator = [&try_write, readable] {
        try_write(readable ? '\n' : ' ', "separator");
    };

    const auto add_soft_separator = [&try_write] {
        try_write(' ', "separator");
    };

    const auto write_matrix = [&](const auto &mat, const auto &what) {
        for (HMM_SIZE_T i {}; i < mat.nrows(); ++i) {
            for (HMM_SIZE_T j {}; j < mat.ncols(); ++j) {
                try_write(mat(i, j), what);
                if (j != mat.ncols() - 1)
                    add_soft_separator();
            }
            if (i != mat.nrows() - 1)
                add_hard_separator();
        }
        add_hard_separator();
    };

    try_write(theta.A.nrows(), "state count");
    add_soft_separator();
    try_write(theta.B.ncols(), "symbol count");
    add_hard_separator();

    write_matrix(theta.A, "transition probability");
    write_matrix(theta.B, "emission probability");

    for (HMM_SIZE_T i {}; i < theta.pi.size(); ++i) {
        try_write(theta.pi(i), "initial distribution value");
        if (i != theta.pi.size() - 1)
            add_soft_separator();
    }
    // Add a trailing newline in case there are multiple models in one stream.
    add_hard_separator();
}

[[nodiscard]]
auto is_probability(HMM_DATA_T p)
{
    return !std::isnan(p) && p >= HMM_DATA_T {} && p <= HMM_DATA_T {1};
}

[[nodiscard]]
auto is_probability_distribution(const data_vector_t &arr)
{
    double accum {};
    for (const auto p: arr) {
        if (!is_probability(p))
            return false;
        accum += p;
    }
    return isclose_rel(accum, HMM_DATA_T {1}, HMM_EPSILON);
}

[[nodiscard]]
auto is_probability_distribution(const data_matrix_t &mat)
{
    for (HMM_SIZE_T i {}; i < mat.nrows(); ++i) {
        double accum {};
        for (HMM_SIZE_T j {}; j < mat.ncols(); ++j) {
            const auto p = mat(i, j);
            if (!is_probability(p))
                return false;
            accum += p;
        }
        if (!isclose_rel(accum, HMM_DATA_T {1}, HMM_EPSILON))
            return false;
    }
    return true;
}

auto check_model_parameters(const model_parameters &theta)
{
    if (!is_probability_distribution(theta.A))
        throw error {"transition matrix contains invalid probabilities"};

    if (!is_probability_distribution(theta.B))
        throw error {"emission matrix contains invalid probabilities"};

    if (!is_probability_distribution(theta.pi))
        throw error {"initial distribution vector contains invalid probabilities"};

    if (theta.A.nrows() != theta.A.ncols())
        throw error {"transmission matrix is not square"};

    if (theta.A.nrows() != theta.B.nrows())
        throw error {"transmission and emission matrices have mismatched first dimensions"};

    if (theta.A.nrows() != theta.pi.size())
        throw error {"initial distribution has incorrect number of states"};
}

auto fill_missing_counts(const size_vector_v &Xs, const size_vector_v &Ys, pseudocounts &counts)
{
    HMM_SIZE_T N {};
    HMM_SIZE_T M {};

    // Try to get the number of allowed states/symbols from the pseudocounts.
    if (counts.B) {
        N = counts.B->nrows();
        M = counts.B->ncols();
    }
    if (counts.A)
        N = counts.A->nrows();
    if (counts.pi)
        N = counts.pi->size();

    // Fill in missing values.
    if (N == 0) {
        for (const auto &Y: Ys) {
            for (const auto y: Y)
                N = std::max(N, y + 1);
        }
    }
    if (M == 0) {
        for (const auto &X: Xs) {
            for (const auto x: X)
                M = std::max(M, x + 1);
        }
    }
    // Allocate missing pseudocounts (set to zeros).
    if (!counts.A)
        counts.A.emplace(N, N);
    if (!counts.B)
        counts.B.emplace(N, M);
    if (!counts.pi)
        counts.pi.emplace(N);
}

[[nodiscard]]
auto estimate(const size_vector_v &Xs, const size_vector_v &Ys, pseudocounts counts) -> model_parameters
{
    fill_missing_counts(Xs, Ys, counts);

    auto transitions = std::move(*counts.A);
    auto emissions = std::move(*counts.B);
    auto initial = std::move(*counts.pi);
    const auto N = emissions.nrows();
    const auto M = emissions.ncols();

    for (HMM_SIZE_T i {}; i < Xs.size(); ++i) {
        const auto &X = Xs[i];
        const auto &Y = Ys[i];
        initial(Y(0))++;
        emissions(Y(0), X(0))++;
        for (HMM_SIZE_T t {1}; t < X.size(); ++t) {
            transitions(Y(t - 1), Y(t))++;
            emissions(Y(t), X(t))++;
        }
    }

    data_matrix_t A {N, N};
    data_matrix_t B {N, M};
    HMM_SIZE_T pi_accum {};

    for (HMM_SIZE_T i {}; i < N; ++i) {
        HMM_SIZE_T accum {};
        for (HMM_SIZE_T j {}; j < N; ++j)
            accum += transitions(i, j);
        for (HMM_SIZE_T j {}; j < N; ++j)
            A(i, j) = HMM_DATA_T(transitions(i, j)) / HMM_DATA_T(accum);
        accum = 0;
        for (HMM_SIZE_T j {}; j < M; ++j)
            accum += emissions(i, j);
        for (HMM_SIZE_T j {}; j < M; ++j)
            B(i, j) = HMM_DATA_T(emissions(i, j)) / HMM_DATA_T(accum);
        pi_accum += initial(i);
    }
    data_vector_t pi {N};
    for (HMM_SIZE_T i {}; i < N; ++i)
        pi(i) = HMM_DATA_T(initial(i)) / HMM_DATA_T(pi_accum);

    batched_log(A);
    batched_log(B);
    batched_log(pi);
    return {A, B, pi};
}

[[nodiscard]]
auto generate(std::default_random_engine &rng, const model_parameters &theta, HMM_SIZE_T T) -> generate_result
{
    const auto &[A, B, pi] = theta;
    std::uniform_real_distribution<HMM_DATA_T> dist;

    const auto choose = [&](HMM_SIZE_T n, const HMM_DATA_T *const p) {
        auto r = dist(rng);
        for (HMM_SIZE_T i {}; i < n; ++i) {
            if ((r -= std::exp(p[i])) <= 0)
                return i;
        }
        // This shouldn't really happen if the row p sums to unity.
        return n - 1;
    };

    hmm::size_vector_t X {T};
    hmm::size_vector_t Y {T};

    for (HMM_SIZE_T t {}; t < T; ++t) {
        Y(t) = choose(A.ncols(), t == 0 ? &pi(0) : &A(Y(t - 1), 0));
        X(t) = choose(B.ncols(), &B(Y(t), 0));
    }
    return {X, Y};
}

[[nodiscard]]
auto compute_log_p(const model &m, const size_vector_v &X) -> HMM_DATA_T
{
    auto log_p = -inf;
    for (const auto &x: X)
        log_p = log_add(log_p, m.evaluate(x).log_p);
    return log_p;
}

model::model(model_parameters theta)
    : m_theta {std::move(theta)}
{
    check_model_parameters(m_theta);

    // Convert to log probabilities.
    batched_log(m_theta.A);
    batched_log(m_theta.B);
    batched_log(m_theta.pi);
}

model::model(std::istream &is)
    : m_theta {load_parameters(is)}
{
    check_model_parameters(m_theta);
    batched_log(m_theta.A);
    batched_log(m_theta.B);
    batched_log(m_theta.pi);
}

model::model(const size_vector_t &X, const size_vector_t &Y)
    : m_theta {estimate({X}, {Y}, {})}
{}

model::model(const size_vector_v &X, const size_vector_v &Y)
    : m_theta {estimate(X, Y, {})}
{}

model::model(const size_vector_t &X, const size_vector_t &Y, const pseudocounts &counts)
    : m_theta {estimate({X}, {Y}, counts)}
{}

model::model(const size_vector_v &X, const size_vector_v &Y, const pseudocounts &counts)
    : m_theta {estimate(X, Y, counts)}
{}

auto model::seed(HMM_SIZE_T seed) -> void
{
    m_rng.seed(seed);
}

auto model::N() const -> HMM_SIZE_T
{
    return m_theta.A.nrows();
}

auto model::M() const -> HMM_SIZE_T
{
    return m_theta.B.ncols();
}

auto model::theta() const -> model_parameters
{
    auto theta = m_theta;

    // Convert back to probabilities.
    batched_exp(theta.A);
    batched_exp(theta.B);
    batched_exp(theta.pi);
    return theta;
}

auto model::generate(HMM_SIZE_T T) -> generate_result
{
    using hmm::generate;
    return generate(m_rng, m_theta, T);
}

auto model::evaluate(const size_vector_t &X) const -> evaluate_result
{
    const auto alpha = forward(X, m_theta);
    auto accum = -inf;

    for (HMM_SIZE_T i {}; i < alpha.ncols(); ++i)
        accum = log_add(accum, alpha(alpha.nrows() - 1, i));
    return {alpha, accum};
}

auto model::predict(const size_vector_t &X) const -> predict_result
{
    auto [delta, Y] = viterbi(X, m_theta);
    normalize_chunks(delta.nrows(), delta.ncols(), &delta(0, 0)); // TODO
    batched_exp(delta);
    return {delta, Y};
}

auto model::decode(const size_vector_t &X) const -> decode_result
{
    // Note that alpha and beta will end up having different scale parameters.
    auto alpha = forward(X, m_theta);
    auto beta = backward(X, m_theta);
    auto gamma = state_posterior(alpha, beta);
    const auto T = alpha.nrows();
    const auto N = alpha.ncols();

    // Normalize each row of alpha and beta.
    normalize_chunks(T, N, &alpha(0, 0));
    normalize_chunks(T, N, &beta(0, 0));

    batched_exp(alpha);
    batched_exp(beta);
    batched_exp(gamma);
    return {alpha, beta, gamma};
}

auto model::fit(const size_vector_v &Xs, const fix_parameters &fixed) -> fit_result
{
    for (const auto &X: Xs) {
        const auto alpha = forward(X, m_theta);
        const auto beta = backward(X, m_theta);
        const auto gamma = state_posterior(alpha, beta);
        const auto xi = segment_posterior(X, m_theta, alpha, beta);

        update_model(X, m_theta, gamma, xi, fixed);
    }
    return {compute_log_p(*this, Xs)};
}

auto model::fit(const size_vector_t &X, const fix_parameters &fixed) -> fit_result
{
    // NOTE: The "size_vector_v" is necessary to prevent infinite recursion.
    return fit(size_vector_v {X}, fixed);
}

auto model::save(std::ostream &os, bool readable) const -> void
{
    auto theta = m_theta;
    batched_exp(theta.A);
    batched_exp(theta.B);
    batched_exp(theta.pi);
    save_parameters(os, theta, readable);
}

} // namespace hmm

