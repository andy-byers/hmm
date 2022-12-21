/*
 * A small hidden Markov model (HMM) library for C++.
 *
 *  Notation
 * ----------
 * N     : Number of allowed states
 * M     : Number of allowed symbols
 * X     : Observed sequence of symbols
 * Y     : Hidden sequence of states associated with X
 * T     : Number of observations/hidden states
 * A     : Transition probability matrix of size (N, N)
 * B     : Emission probability matrix of size (N, M)
 * pi    : Initial state distribution
 * theta : HMM consisting of (A, B, pi)
 * alpha : Forward probability matrix of size (T, N)
 * beta  : Backward probability matrix of size (T, N)
 * gamma : Posterior probability matrix of size (T, N)
 *
 *  Notes
 * -------
 * 1. The model class expects and emits normal probabilities (not log probabilities) unless
 *    otherwise noted. The A and B parameters should have rows that sum to roughly 1.0. The
 *    alpha, beta, and gamma parameters will also have normalized rows.
 *
 *  References
 * ------------
 * See README.md
 */

#ifndef HMM_HMM_H
#define HMM_HMM_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <list>
#include <optional>
#include <random>
#include <string>
#include <vector>

#ifdef HMM_SMALL
#  define HMM_DATA_T float
#  define HMM_SIZE_T std::uint32_t
#else
#  define HMM_DATA_T double
#  define HMM_SIZE_T std::uint64_t
#endif

#define HMM_ASSERT assert
#define HMM_EPSILON (HMM_DATA_T {1e-3})

namespace hmm {

template<std::floating_point T>
auto isclose_abs(T x, T y, T epsilon)
{
    return (x > y ? x - y : y - x) <= epsilon;
}

template<std::floating_point T>
auto isclose_rel(T x, T y, T epsilon)
{
    epsilon *= std::max(std::abs(x), std::abs(y));
    return isclose_abs(x, y, epsilon);
}

class error: public std::exception {
public:
    explicit error(std::string what)
        : m_what {std::move(what)}
    {}

    [[nodiscard]]
    auto what() const noexcept -> const char * override
    {
        return m_what.c_str();
    }

private:
    std::string m_what;
};

template<std::regular T>
class vector {
public:
    using value_type = T;
    using inner_type = std::vector<T>;

    constexpr explicit vector(HMM_SIZE_T size)
        : m_data(size)
    {}

    constexpr explicit vector(inner_type data)
        : m_data {std::move(data)}
    {}

    [[nodiscard]]
    constexpr auto size() const -> HMM_SIZE_T
    {
        return m_data.size();
    }

    [[nodiscard]]
    constexpr auto operator()(HMM_SIZE_T i) -> T&
    {
        HMM_ASSERT(i < m_data.size());
        return m_data[i];
    }

    [[nodiscard]]
    constexpr auto operator()(HMM_SIZE_T i) const -> const T&
    {
        HMM_ASSERT(i < m_data.size());
        return m_data[i];
    }

    [[nodiscard]]
    constexpr auto begin() -> typename inner_type::iterator
    {
        using std::begin;
        return begin(m_data);
    }

    [[nodiscard]]
    constexpr auto end() -> typename inner_type::iterator
    {
        using std::end;
        return end(m_data);
    }

    [[nodiscard]]
    constexpr auto begin() const -> typename inner_type::const_iterator
    {
        using std::begin;
        return begin(m_data);
    }

    [[nodiscard]]
    constexpr auto end() const -> typename inner_type::const_iterator
    {
        using std::end;
        return end(m_data);
    }

private:
    inner_type m_data;
};

template<std::regular T>
class matrix: public vector<T> {
public:
    using value_type = typename vector<T>::value_type;
    using inner_type = typename vector<T>::inner_type;

    constexpr matrix(HMM_SIZE_T nrows, HMM_SIZE_T ncols)
        : base_type {nrows * ncols},
          m_ncols {ncols}
    {}

    constexpr explicit matrix(HMM_SIZE_T ncols, inner_type data)
        : base_type {std::move(data)},
          m_ncols {ncols}
    {}

    [[nodiscard]]
    constexpr auto nrows() const -> HMM_SIZE_T
    {
        return base_type::size() / m_ncols;
    }

    [[nodiscard]]
    constexpr auto ncols() const -> HMM_SIZE_T
    {
        return m_ncols;
    }

    [[nodiscard]]
    constexpr auto operator()(HMM_SIZE_T i, HMM_SIZE_T j) -> value_type&
    {
        HMM_ASSERT(i < nrows());
        HMM_ASSERT(j < m_ncols);
        return base_type::operator()(i*m_ncols + j);
    }

    [[nodiscard]]
    constexpr auto operator()(HMM_SIZE_T i, HMM_SIZE_T j) const -> const value_type&
    {
        HMM_ASSERT(i < nrows());
        HMM_ASSERT(j < m_ncols);
        return base_type::operator()(i*m_ncols + j);
    }

private:
    using base_type = vector<T>;
    HMM_SIZE_T m_ncols;
};

using data_matrix_t = matrix<HMM_DATA_T>;
using size_matrix_t = matrix<HMM_SIZE_T>;
using data_vector_t = vector<HMM_DATA_T>;
using size_vector_t = vector<HMM_SIZE_T>;

using data_matrix_v = std::vector<matrix<HMM_DATA_T>>;
using size_matrix_v = std::vector<matrix<HMM_SIZE_T>>;
using data_vector_v = std::vector<vector<HMM_DATA_T>>;
using size_vector_v = std::vector<vector<HMM_SIZE_T>>;

struct model_parameters {
   data_matrix_t A;
   data_matrix_t B;
   data_vector_t pi;
};

struct generate_result {
    size_vector_t X;
    size_vector_t Y;
};

struct evaluate_result {
    data_matrix_t alpha;
    HMM_DATA_T log_p {};
};

struct predict_result {
   data_matrix_t delta;
   size_vector_t Y;
};

struct decode_result {
    data_matrix_t alpha;
    data_matrix_t beta;
    data_matrix_t gamma;
};

struct fix_parameters {
    enum: HMM_SIZE_T {
        TRANSITION = 0,
        EMISSION = 1,
        INITIAL = 2,
    };

    bool values[3] {};
};

struct fit_result {
    HMM_DATA_T log_p {};
};

struct pseudocounts {
    std::optional<size_matrix_t> A;
    std::optional<size_matrix_t> B;
    std::optional<size_vector_t> pi;
};

/*
 * Representation of a HMM.
 */
class model {
public:
    /*
     * Load the model from a stream.
     */
    explicit model(std::istream &is);

    /*
     * Create a model using existing parameters.
     */
    explicit model(model_parameters theta);

    /*
     * Estimate a model from training examples.
     */
    explicit model(const size_vector_t &X, const size_vector_t &Y);
    explicit model(const size_vector_v &X, const size_vector_v &Y);
    explicit model(const size_vector_t &X, const size_vector_t &Y, const pseudocounts &counts);
    explicit model(const size_vector_v &X, const size_vector_v &Y, const pseudocounts &counts);

    /*
     * Set the PRNG seed (used for generating observations/states).
     */
    auto seed(HMM_SIZE_T seed) -> void;

    /*
     * Get the number of allowed states.
     */
    [[nodiscard]] auto N() const -> HMM_SIZE_T;

    /*
     * Get the number of allowed symbols.
     */
    [[nodiscard]] auto M() const -> HMM_SIZE_T;

    /*
     * Get the model parameters: theta = (A, B, pi).
     */
    [[nodiscard]] auto theta() const -> model_parameters;

    /*
     * Generate a sequence of observations, as well as the associated sequence of hidden states.
     */
    [[nodiscard]] auto generate(HMM_SIZE_T T) -> generate_result;

    /*
     * Determine the probability of the observations given the model.
     */
    [[nodiscard]] auto evaluate(const size_vector_t &X) const -> evaluate_result;

    /*
     * Determine the most-likely state sequence for the observations.
     */
    [[nodiscard]] auto predict(const size_vector_t &X) const -> predict_result;

    /*
     * Determine the forward, backward, and posterior probabilities.
     */
    [[nodiscard]] auto decode(const size_vector_t &X) const -> decode_result;

    /*
     * Tune the model parameters using one iteration of Baum-Welch.
     */
    auto fit(const size_vector_t &X, const fix_parameters &fixed = {}) -> fit_result;
    auto fit(const size_vector_v &X, const fix_parameters &fixed = {}) -> fit_result;

    /*
     * Save the model to a stream.
     */
    auto save(std::ostream &os, bool readable = false) const -> void;

private:
    model_parameters m_theta;
    std::default_random_engine m_rng;
};

} // namespace hmm

#endif // HMM_HMM_H