#include "../include/hmm.h"
#include <sstream>

static hmm::model uniform_example {{
    hmm::data_matrix_t {2, {
        0.5, 0.5,
        0.5, 0.5,
    }},
    hmm::data_matrix_t {3, {
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
        1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
    }},
    hmm::data_vector_t {{
        0.5,
        0.5,
    }},
}};

/*
 * Example from https://en.wikipedia.org/wiki/Viterbi_algorithm.
 */
static hmm::model wiki_example {{
    hmm::data_matrix_t {2, {
        0.7, 0.3,
        0.4, 0.6,
    }},
    hmm::data_matrix_t {3, {
        0.5, 0.4, 0.1,
        0.1, 0.3, 0.6,
    }},
    hmm::data_vector_t {{
        0.6,
        0.4,
    }},
}};

static hmm::model visible_example {{
    hmm::data_matrix_t {2, {
        0.5, 0.5,
        0.5, 0.5,
    }},
    // Emissions are uniquely determined by state. Not really a HMM.
    hmm::data_matrix_t {2, {
        1.0, 0.0,
        0.0, 1.0,
    }},
    hmm::data_vector_t {{
        0.5,
        0.5,
    }},
}};

template<template<class> class T>
auto assert_equal(const T<hmm::size_t> &lhs, const T<hmm::size_t> &rhs)
{
    auto a = lhs.begin();
    auto b = rhs.begin();
    while (a != lhs.end() && b != rhs.end())
        HMM_ASSERT(*a++ == *b++);
    HMM_ASSERT(a == lhs.end());
    HMM_ASSERT(b == rhs.end());
}

template<template<class> class T>
auto assert_equal(const T<hmm::data_t> &lhs, const T<hmm::data_t> &rhs, hmm::data_t tol = 1e-3)
{
    auto a = lhs.begin();
    auto b = rhs.begin();
    while (a != lhs.end() && b != rhs.end())
        HMM_ASSERT(hmm::isclose_abs(*a++, *b++, tol));
    HMM_ASSERT(a == lhs.end());
    HMM_ASSERT(b == rhs.end());
}

auto assert_equal(const hmm::model &lhs, const hmm::model &rhs, hmm::data_t tol = 1e-3)
{
    const auto lhs_theta = lhs.theta();
    const auto rhs_theta = rhs.theta();
    assert_equal(lhs_theta.A, rhs_theta.A, tol);
    assert_equal(lhs_theta.B, rhs_theta.B, tol);
    // pi is usually inaccurate. We would need a ton of training examples to estimate it well.
}

auto test_predict_wiki()
{
    using namespace hmm;
    auto model = wiki_example;
    size_vector_t X {{
        0, 1, 2,
    }};
    const auto [delta, path] = model.predict(X);

    HMM_ASSERT(path.size() == 3);
    HMM_ASSERT(path(0) == 0);
    HMM_ASSERT(path(1) == 0);
    HMM_ASSERT(path(2) == 1);
    HMM_ASSERT(delta.nrows() == 3);
    HMM_ASSERT(delta.ncols() == 2);

    // deltas are normalized before they are converted back from log space to avoid underflow.
    static constexpr hmm::data_t eps_factor {1e-5};
    HMM_ASSERT(isclose_abs(delta(0, 0), 0.3 / (0.3+0.04), eps_factor));
    HMM_ASSERT(isclose_abs(delta(0, 1), 0.04 / (0.3+0.04), eps_factor));
    HMM_ASSERT(isclose_abs(delta(1, 0), 0.084 / (0.084+0.027), eps_factor));
    HMM_ASSERT(isclose_abs(delta(1, 1), 0.027 / (0.084+0.027), eps_factor));
    HMM_ASSERT(isclose_abs(delta(2, 0), 0.00588 / (0.00588+0.01512), eps_factor));
    HMM_ASSERT(isclose_abs(delta(2, 1), 0.01512 / (0.00588+0.01512), eps_factor));
}

auto test_predict_matlab()
{
    using namespace hmm;

    data_matrix_t A {3, {
        0.6, 0.2, 0.2,
        0.2, 0.6, 0.2,
        0.2, 0.2, 0.6,
    }};
    data_matrix_t B {4, {
        0.5, 0.1, 0.2, 0.2,
        0.1, 0.5, 0.3, 0.1,
        0.2, 0.3, 0.1, 0.4,
    }};
    data_vector_t pi {{
        1.0, 0.0, 0.0,
    }};

    size_vector_t X {{ 0, 1, 0, 2, 3, 2, 0, 3, 1, 0, 0, 2, 2, 3, 0, 0, 0, 3, 1, 1, 0, 2, 1, 1, 3, 3, 3, 0, 1, 3, 1, 3, 3, 2, 1, 0, 1, 0, 1, 3, 0, 0, 0, 3, 2, 1, 1, 3, 2, 2, 3, 1, 2, 1, 2, 1, 2, 3, 2, 1, 1, 0, 0, 0, 0, 0, 3, 1, 1, 2, 1, 2, 0, 1, 2, 3, 0, 3, 3, 0, 1, 0, 3, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 0, 1, 3, 0, 3, 1, 0, 0, 3, 0, 3, 0, 0, 3, 0, 2, 3, 3, 3, 2, 1, 1, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 3, 0, 3, 0, 1, 0, 0, 2, 2, 3, 2, 3, 2, 3, 2, 1, 3, 1, 2, 0, 3, 0, 0, 2, 3, 3, 1, 3, 1, 1, 3, 3, 0, 3, 1, 1, 1, 2, 3, 2, 2, 3, 2, 1, 0, 3, 1, 1, 1, 1, 1, 3, 1, 0, 2, 0, 0, 1, 2, 0, 1, 0, 1, 1, 3, 1, 2, 1, 2, 2, 1, 0, 0, 3, 1, 0, 1, 2, 2, 2, 0, 1, 3, 1, 2, 1, 0, 1, 3, 3, 2, 0, 0, 0, 0, 3, 1, 2, 2, 0, 3, 0, 2, 2, 2, 1, 1, 1, 1, 2, 2, 3, 1, 3, 0, 1, 1, 1, 3, 1}};
    size_vector_t answer {{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2}};

    model model {{A, B, pi}};

    const auto [delta, path] = model.predict(X);
    for (hmm::size_t i {}; i < answer.size(); ++i)
        HMM_ASSERT(path(i) == answer(i));
}

auto test_predict()
{
    test_predict_wiki();
    test_predict_matlab();
}

auto test_decode_uniform()
{
    auto model = uniform_example;
    hmm::size_vector_t X{{0, 0, 0}};
    const auto [alpha, beta, gamma] = model.decode(X);
    static constexpr hmm::data_t eps {1e-8};

    for (const auto a: alpha)
        HMM_ASSERT(hmm::isclose_abs(a, 0.5, eps));
    for (const auto b: beta)
        HMM_ASSERT(hmm::isclose_abs(b, 0.5, eps));
    for (const auto g: gamma)
        HMM_ASSERT(hmm::isclose_abs(g, 0.5, eps));
}

auto test_decode_visible()
{
    auto model = visible_example;
    hmm::size_vector_t X{{0, 1, 0, 1}};
    const auto [alpha, beta, gamma] = model.decode(X);
    static constexpr hmm::data_t eps {1e-8};

    HMM_ASSERT(hmm::isclose_abs(alpha(0, 0), 1.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(0, 1), 0.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(1, 0), 0.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(1, 1), 1.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(2, 0), 1.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(2, 1), 0.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(3, 0), 0.0, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(3, 1), 1.0, eps));

    HMM_ASSERT(hmm::isclose_abs(beta(0, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(0, 1), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(1, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(1, 1), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(2, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(2, 1), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(3, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(beta(3, 1), 0.5, eps));

    HMM_ASSERT(hmm::isclose_abs(gamma(0, 0), alpha(0, 0), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(0, 1), alpha(0, 1), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(1, 0), alpha(1, 0), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(1, 1), alpha(1, 1), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(2, 0), alpha(2, 0), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(2, 1), alpha(2, 1), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(3, 0), alpha(3, 0), eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(3, 1), alpha(3, 1), eps));
}

// NOTE: MATLAB uses a slightly different algorithm for decoding that doesn't have an initial distribution.
//       We can simulate an initial distribution in MATLAB by explicitly representing the "Begin" state in
//       the A and B matrices.
auto test_decode_matlab()
{
    hmm::model model {{
        hmm::data_matrix_t {2, {
            0.75, 0.25,
            0.25, 0.75,
        }},
        hmm::data_matrix_t {2, {
            0.25, 0.75,
            0.75, 0.25,
        }},
        hmm::data_vector_t {{
            0.5,
            0.5,
        }},
    }};
    hmm::size_vector_t X {{0, 1, 0, 1, 0}};
    const auto [alpha, beta, gamma] = model.decode(X);
    static constexpr hmm::data_t eps {1e-4};

    HMM_ASSERT(hmm::isclose_abs(alpha(0, 0), 0.2500, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(0, 1), 0.7500, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(1, 0), 0.6429, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(1, 1), 0.3571, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(2, 0), 0.3077, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(2, 1), 0.6923, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(3, 0), 0.6702, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(3, 1), 0.3298, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(4, 0), 0.3198, eps));
    HMM_ASSERT(hmm::isclose_abs(alpha(4, 1), 0.6802, eps));

    // MATLAB seems to adjust the backward probabilities such that the first one is 1.0. Anyway, the ratios should be the same.
    HMM_ASSERT(hmm::isclose_abs(beta(0, 0), 1.2791 / (1.2791+0.9070), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(0, 1), 0.9070 / (1.2791+0.9070), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(1, 0), 0.8547 / (0.8547+1.2616), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(1, 1), 1.2616 / (0.8547+1.2616), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(2, 0), 1.2093 / (1.2093+0.9070), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(2, 1), 0.9070 / (1.2093+0.9070), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(3, 0), 0.8198 / (0.8198+1.3663), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(3, 1), 1.3663 / (0.8198+1.3663), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(4, 0), 1.0000 / (1.0000+1.0000), eps));
    HMM_ASSERT(hmm::isclose_abs(beta(4, 1), 1.0000 / (1.0000+1.0000), eps));

    HMM_ASSERT(hmm::isclose_abs(gamma(0, 0), 0.3198, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(0, 1), 0.6802, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(1, 0), 0.5494, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(1, 1), 0.4506, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(2, 0), 0.3721, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(2, 1), 0.6279, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(3, 0), 0.5494, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(3, 1), 0.4506, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(4, 0), 0.3198, eps));
    HMM_ASSERT(hmm::isclose_abs(gamma(4, 1), 0.6802, eps));
}

auto test_decode()
{
    test_decode_uniform();
    test_decode_visible();
    test_decode_matlab();
}

auto test_estimate_wiki()
{
    auto source = wiki_example;
    hmm::size_vector_v wiki_Xs;
    hmm::size_vector_v wiki_Ys;
    for (hmm::size_t i {}; i < 100; ++i) {
        auto [X, Y] = source.generate(100);
        wiki_Xs.emplace_back(X);
        wiki_Ys.emplace_back(Y);
    }

    const auto model = hmm::model {wiki_Xs, wiki_Ys};
    const auto wiki_theta = wiki_example.theta();
    const auto theta = model.theta();

    assert_equal(theta.A, wiki_theta.A, 0.1);
    assert_equal(theta.B, wiki_theta.B, 0.1);
}

auto test_estimate_pseudocounts()
{
//    auto source = wiki_example;
//    hmm::size_vector_t X {{
//        0, 1, 1, 0,
//    }};
//    hmm::size_vector_t Y {{
//        1, 0, 0, 1,
//    }};
//
//    hmm::pseudocounts pseudocounts;
//    pseudocounts.A = hmm::data_matrix_t {1, 1};
//
//    const auto model = hmm::model {wiki_Xs, wiki_Ys};
//    const auto wiki_theta = wiki_example.theta();
//    const auto theta = model.theta();
//
//    assert_equal(theta.A, wiki_theta.A, 0.1);
//    assert_equal(theta.B, wiki_theta.B, 0.1);
}

auto test_estimate()
{
    test_estimate_wiki();
    test_estimate_pseudocounts();
}

auto test_fit(const hmm::size_vector_v &Xs, const hmm::size_vector_v &)
{
    using namespace hmm;

    model model {{
        data_matrix_t {3, {
            1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
            1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
            1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
        }},
        data_matrix_t {4, {
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
        }},
        data_vector_t {{
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        }},
    }};
    for (hmm::size_t iteration {}; iteration < 10; ++iteration)
        model.fit(Xs);

    const auto theta = model.theta();
    data_matrix_t answer_A {3, {
        0.6, 0.2, 0.2,
        0.2, 0.6, 0.2,
        0.2, 0.2, 0.6,
    }};
    data_matrix_t answer_B {4, {
        0.5, 0.1, 0.2, 0.2,
        0.1, 0.5, 0.3, 0.1,
        0.2, 0.3, 0.1, 0.4,
    }};
}

auto test_fit_uniform()
{
    auto source = uniform_example;
    auto target = uniform_example;

    // Each symbol has equal representation. With uniform priors, nothing should change.
    hmm::size_vector_t X {{0, 1, 2, 0, 1, 2}};

    for (hmm::size_t iteration {}; iteration < 100; ++iteration)
        source.fit(X);

    assert_equal(source, target);
}

auto test_fit_nonuniform()
{
    auto model = uniform_example;

    // This time, symbol 0 appears more often than the other symbols. p(0) = 2 / 3,
    // p(1) = 1 / 6, and p(2) = 1 / 6.
    hmm::size_vector_t X {{0, 1, 2, 0, 0, 0}};

    for (hmm::size_t iteration {}; iteration < 100; ++iteration)
        model.fit(X);

    const auto [A, B, pi] = model.theta();
    static constexpr hmm::data_t eps {1e-5};

    HMM_ASSERT(hmm::isclose_abs(A(0, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(A(0, 1), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(A(1, 0), 0.5, eps));
    HMM_ASSERT(hmm::isclose_abs(A(1, 1), 0.5, eps));

    HMM_ASSERT(hmm::isclose_abs(B(0, 0), 2.0 / 3.0, eps));
    HMM_ASSERT(hmm::isclose_abs(B(0, 1), 1.0 / 6.0, eps));
    HMM_ASSERT(hmm::isclose_abs(B(0, 2), 1.0 / 6.0, eps));
    HMM_ASSERT(hmm::isclose_abs(B(1, 0), 2.0 / 3.0, eps));
    HMM_ASSERT(hmm::isclose_abs(B(1, 1), 1.0 / 6.0, eps));
    HMM_ASSERT(hmm::isclose_abs(B(1, 2), 1.0 / 6.0, eps));
}

auto test_fit()
{
    test_fit_uniform();
    test_fit_nonuniform();
}

auto test_generate()
{
    auto model = wiki_example;

    // We need to generate enough examples to get a decent estimate of the parameters.
    hmm::size_vector_v Xs;
    hmm::size_vector_v Ys;
    for (hmm::size_t n {}; n < 100; ++n) {
        auto [X, Y] = model.generate(256);
        Xs.emplace_back(X);
        Ys.emplace_back(Y);
    }

    // Try to estimate the parameters we used to generate the state/observation sequences.
    assert_equal(hmm::model {Xs, Ys}, model, 0.05);
}

auto test_persist_one()
{
    const auto source = wiki_example;

    std::ostringstream oss;
    source.save(oss);

    std::istringstream iss {oss.str()};
    const hmm::model target {iss};

    assert_equal(source, target);
}

// We should actually be able to save/load any number of models to/from the same stream.
auto test_persist_several()
{
    const auto source_1 = wiki_example;
    const auto source_2 = uniform_example;
    const auto source_3 = visible_example;

    std::ostringstream oss;
    source_1.save(oss);
    source_2.save(oss);
    source_3.save(oss);

    std::istringstream iss {oss.str()};
    const hmm::model target_1 {iss};
    const hmm::model target_2 {iss};
    const hmm::model target_3 {iss};

    assert_equal(source_1, target_1);
    assert_equal(source_2, target_2);
    assert_equal(source_3, target_3);
}

auto test_persist_format()
{
    // NOTE: Should be able to handle multiple spaces/newlines, etc.
    const auto readable =
        "2  3\t\t"
        "0.7 0.3  "
        "0.4 0.6\n\n"
        "0.5 0.4 0.1\t"
        "0.1 0.3 0.6\n"
        "0.6 0.4 1.0\n "; // Extra 1.0 shouldn't be read.

    std::istringstream iss {readable};
    assert_equal(hmm::model {iss}, wiki_example, 1e-10);
}

auto test_persist_readable()
{
    // Use values that can be exactly represented. These happen to be integers, and are written
    // without the dot by hmm::model::save().
    const auto readable =
        "2 3\n"
        "1 0\n"
        "0 1\n"
        "1 0 0\n"
        "0 1 0\n"
        "1 0\n";

    std::istringstream iss {readable};
    hmm::model model {iss};

    std::ostringstream oss;
    model.save(oss, true);

    HMM_ASSERT(oss.str() == readable);
}

auto test_persist()
{
    test_persist_one();
    test_persist_several();
    test_persist_format();
    test_persist_readable();
}

auto main(int, char**) -> int
{
    uniform_example.seed(42);
    visible_example.seed(42);
    wiki_example.seed(42);

    test_generate();
    test_predict();
    test_decode();
    test_persist();
    test_estimate();
    test_fit();
    return 0;
}