/*
 * hmm: A small hidden Markov model (HMM) library for C++.
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

#define HMM_ASSERT assert
#define HMM_EPSILON 1e-3

namespace hmm {

using size_t = std::size_t;
using data_t = double;

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

    constexpr explicit vector(size_t size)
        : m_data(size)
    {}

    constexpr explicit vector(inner_type data)
        : m_data {std::move(data)}
    {}

    [[nodiscard]]
    constexpr auto size() const -> size_t
    {
        return m_data.size();
    }

    [[nodiscard]]
    constexpr auto operator()(size_t i) -> T&
    {
        HMM_ASSERT(i < m_data.size());
        return m_data[i];
    }

    [[nodiscard]]
    constexpr auto operator()(size_t i) const -> const T&
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

    constexpr matrix(size_t nrows, size_t ncols)
        : base_type {nrows * ncols},
          m_ncols {ncols}
    {}

    constexpr explicit matrix(size_t ncols, inner_type data)
        : base_type {std::move(data)},
          m_ncols {ncols}
    {}

    [[nodiscard]]
    constexpr auto nrows() const -> size_t
    {
        return base_type::size() / m_ncols;
    }

    [[nodiscard]]
    constexpr auto ncols() const -> size_t
    {
        return m_ncols;
    }

    [[nodiscard]]
    constexpr auto operator()(size_t i, size_t j) -> value_type&
    {
        HMM_ASSERT(i < nrows());
        HMM_ASSERT(j < m_ncols);
        return base_type::operator()(i*m_ncols + j);
    }

    [[nodiscard]]
    constexpr auto operator()(size_t i, size_t j) const -> const value_type&
    {
        HMM_ASSERT(i < nrows());
        HMM_ASSERT(j < m_ncols);
        return base_type::operator()(i*m_ncols + j);
    }

private:
    using base_type = vector<T>;
    size_t m_ncols;
};

using data_matrix_t = matrix<data_t>;
using size_matrix_t = matrix<size_t>;
using data_vector_t = vector<data_t>;
using size_vector_t = vector<size_t>;

using data_matrix_v = std::vector<matrix<data_t>>;
using size_matrix_v = std::vector<matrix<size_t>>;
using data_vector_v = std::vector<vector<data_t>>;
using size_vector_v = std::vector<vector<size_t>>;

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
    data_t log_p {};
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
    enum: size_t {
        TRANSITION = 0,
        EMISSION = 1,
        INITIAL = 2,
    };

    bool values[3] {};
};

struct fit_result {
    data_t log_p {};
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
    auto seed(size_t seed) -> void;

    /*
     * Get the number of allowed states.
     */
    [[nodiscard]] auto N() const -> size_t;

    /*
     * Get the number of allowed symbols.
     */
    [[nodiscard]] auto M() const -> size_t;

    /*
     * Get the model parameters: theta = (A, B, pi).
     */
    [[nodiscard]] auto theta() const -> model_parameters;

    /*
     * Generate a sequence of observations, as well as the associated sequence of hidden states.
     */
    [[nodiscard]] auto generate(size_t T) -> generate_result;

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