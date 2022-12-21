# hmm
`hmm` is a small C++ library for working with hidden Markov models (HMMs).

## Description
We represent a HMM with 3 model parameters: $A$, $B$, and $\pi$.
$A$ represents the transition probabilities, where $A_{ij} = P(Y_{t+1}=j|Y_t=i)$ is the probability of transitioning into state $j$ from state $i$.
$A$ is of size $N\times N$, where $N$ is the number of available states.
$B$ represents the emission probabilities, where $B_j(k) = P(x_t=k|y_t=j)$ is the probability of emitting symbol $k$ from state $j$.
$B$ is of size $N\times M$, where $M$ is the number of available symbols.
Finally, $\pi$ represents the initial state distribution, where $\pi_{i}$ is the probability of starting in state $i$.
$pi$ contains an element for each of the $N$ states.

`hmm` uses the `hmm::model` class to represent a HMM.
`hmm::model` can be constructed in several ways:
1. Model parameters $\theta = (A, B, \pi)$ supplied as `hmm::model_parameters`
2. Parameters read as text from a `std::istream` (see [Text format](#text-format))
3. Parameters estimated from training examples

Once we have a `hmm::model`, we can use it to:
1. Generate observation and state sequences given the model parameters
2. Decode the forward, backward, and posterior probabilities
3. Predict the most-likely state sequence corresponding to the observations (Viterbi)
4. Train the model using example observations (Baum-Welch)

## Features
+ Performs calculations in log space to avoid floating-point underflow
+ Supports pseudocounts to avoid zero probabilities during parameter estimation
+ Parameters can be fixed during training
+ Serialization of models

## Caveats
+ Uses dense data structures to represent the model parameters
  + Causes wasted space when there are relatively few possible transitions
  + Makes this implementation unfeasible for high-order models

## Text format
`hmm` supports a simple serialization protocol.
The model parameters, along with size parameters $N$ and $M$, are converted into text using C++ Standard Library functions and concatenated in the following order:
```
N M A B pi
```

The spaces can be replaced with any number of whitespace characters, including newlines, and multiple models can be saved to the same stream.
For serialization and deserialization, respectively, `hmm::model` provides a `save(std::ostream &)` method, and a constructor that takes a `std::istream &`.
Here's an example of what this text format might look like for a model with 2 states and 3 symbols. 
Note that each data element $p$ is a probability ($0\le p\le1$) and each row a discrete probability distribution ($\sum_j p_j = 1$).
```
2 3

0.8 0.2
0.2 0.8

0.5 0.4 0.1
0.4 0.5 0.1

0.5 0.5
```

## TODO
+ Not sure if the initial distribution update is correct, needs to be tested
+ More testing
+ Examples/use cases
+ Sparse matrices for handling large models (higher-order models converted to first-order with a ton of states)

## References
1. Durbin, R., Eddy, S., Krogh, A., and Mitchison, G. (2010). Biological Sequence Analysis.
2. MATLAB. (2010). version 7.10.0 (R2010a). Natick, Massachusetts: The MathWorks Inc.
3. Hasegawa-Johnson, M. (2020). ECE 417: Multimedia Signal Processing. The Grainger College of Engineering. Retrieved December 21, 2022, from https://courses.engr.illinois.edu/ece417/fa2020/
4. Wikimedia Foundation. (2022). List of logarithmic identities. Wikipedia. Retrieved December 21, 2022, from https://en.wikipedia.org/wiki/List_of_logarithmic_identities 