# Activation Function Selection Guide

Choosing the right activation function is critical for nonlinear regression tasks, particularly when modelling complex interactions across wildly different feature distributions.

## Available Activations

### Sigmoid
- **Formula**: $f(x) = \frac{1}{1 + e^{-x}}$
- **Properties**: Outputs bounded between (0, 1). High risk of vanishing gradients in deep networks.
- **Best Use Case**: Output layers for probability, bounded regression targets.

### Tanh
- **Formula**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **Properties**: Outputs bounded between (-1, 1). Stronger gradients than sigmoid, but still suffers vanishing gradients for large inputs.
- **Best Use Case**: Centered output requirements.

### ReLU (Rectified Linear Unit)
- **Formula**: $f(x) = \max(0, x)$
- **Properties**: Fast, standard. Risk of "dead neurons" (neurons stuck at 0 gradient).
- **Best Use Case**: Shallow networks, fast prototyping.

### LeakyReLU / PReLU
- **Formula**: $f(x) = \max(\alpha x, x)$ (where $\alpha$ is small fixed, or learnable in PReLU)
- **Properties**: Solves the dead neuron problem. PReLU offers maximum gradient flow at the cost of parameters.
- **Best Use Case**: When ReLU exhibits dead neurons, or maximum gradient flow is needed.

### GELU (Gaussian Error Linear Unit)
- **Formula**: $f(x) = x \cdot \Phi(x)$ (where $\Phi(x)$ is the standard Gaussian cumulative distribution)
- **Properties**: Smooth nonlinearity, acts as a probabilistic dropout.
- **Best Use Case**: Deep networks, NLP, standardizing variance.

### SiLU (Swish)
- **Formula**: $f(x) = x \cdot \sigma(x)$
- **Properties**: Smooth, non-monotonic. Does not suffer from dead neurons.
- **Best Use Case**: General purpose regression, deep networks.

### Mish
- **Formula**: $f(x) = x \cdot \tanh(\text{softplus}(x))$
- **Properties**: Similar to SiLU but with steeper negative gradients. Heavy computation.
- **Best Use Case**: Highly complex topologies where gradient flow is critical.

## Selection Criteria

Activation choice in this project primarily affects:
1. **Convergence speed**: ReLU/LeakyReLU converge quickly; Mish/GELU require more epochs but might find better minima.
2. **Dead neuron percentage**: Heavily impactful in our heterogeneous dataset; negative-allowing activations (SiLU, GELU, Mish) prevent this.
3. **Gradient flow through deep networks**: Smooth, non-monotonic functions propagate gradients better over >5 layers.
4. **Interaction with BatchNorm**: Some activations (like GELU) self-regularize and may conflict or synergize unexpectedly with BN.
5. **Computational overhead**: ReLU is virtually free; Mish requires exponentials and division, slowing down training.

## Recommendation Matrix

| Scenario | Recommended | Rationale |
|----------|------------|----------|
| **Default / general purpose** | **SiLU (Swish)** | Smooth, non-monotonic, excellent gradient flow |
| Very deep networks (>8 layers) | GELU or Mish | Built-in regularization effect, entirely avoids dead neurons |
| Shallow networks (<4 layers) | ReLU or LeakyReLU | Fast, mathematically simple, sufficient for low depth |
| Output bounded [0,1] | Sigmoid | Natural probability or normalized target output |
| Output bounded [-1,1] | Tanh | Centered target output |
| Maximum gradient flow | PReLU | Learnable negative slope adapts to dataset |

## Why SiLU for This Project?
**SiLU (Swish)** was selected as the default activation function for the MPG prediction task. The synthetic dataset contains high-variance, heteroscedastic noise and interacting non-linear features (e.g. thermodynamics cycles and logarithms). A smooth, non-monotonic activation like SiLU inherently handles these smooth polynomial and exponential interactions better than piecewise functions like ReLU, while avoiding the dead-neuron collapse that can happen when wide variance features hit a ReLU layer negatively.
