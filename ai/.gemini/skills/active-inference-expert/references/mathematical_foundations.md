# Mathematical Foundations of Active Inference

Active Inference is grounded in several mathematical disciplines, primarily information theory, variational Bayes, and non-equilibrium statistical mechanics.

## Variational Free Energy (VFE)

Variational free energy is the core objective function in Active Inference. It is an upper bound on surprise (negative log evidence).

### Formal Definition
$$F[q] = \mathbb{E}_q[\ln q(s) - \ln p(s,o)]$$

Where:
- $q(s)$ is the recognition density (the agent's internal belief about hidden states $s$).
- $p(s,o)$ is the generative model relating hidden states $s$ to observations $o$.

### Decompositions
VFE can be decomposed into different components that reveal the trade-offs in inference:

1. **Accuracy and Complexity**:
   $$F[q] = \underbrace{\mathbb{E}_q[-\ln p(o|s)]}_{	ext{Accuracy}} + \underbrace{D_{\mathrm{KL}}[q(s) \| p(s)]}_{	ext{Complexity}}$$
   Minimizing free energy involves maximizing accuracy (model fit) while minimizing complexity (deviation from priors).

2. **Divergence and Evidence**:
   $$F[q] = D_{\mathrm{KL}}[q(s) \| p(s|o)] - \ln p(o)$$
   Since KL divergence is always non-negative, $F \geq -\ln p(o)$ (Surprise).

## Expected Free Energy (EFE)

While VFE is used for perception (inference about the present), Expected Free Energy is used for action (policy selection for the future).

$$G(\pi, 	au) = \mathbb{E}_{q(o_	au, s_	au|\pi)}[\ln q(s_	au|\pi) - \ln p(o_	au, s_	au)]$$

EFE includes:
- **Pragmatic Value**: Realizing goals/preferences.
- **Epistemic Value**: Reducing uncertainty through exploration.

## Markov Blankets

The Markov blanket defines the boundaries of an agent. It consists of:
- **Sensory States**: Information flowing from external to internal.
- **Active States**: Information flowing from internal to external.
- **Internal States**: The agent's brain or state.
- **External States**: The world outside.

A system with internal states $\mu$ and external states $\eta$ has a Markov blanket $b$ such that:
$$p(\mu | \eta, b) = p(\mu | b)$$

## Implementation Strategies

### Variational Inference
Optimization of the recognition density $q(s)$ often uses gradient descent or message passing on factor graphs.

### Natural Gradients
Using the Fisher Information Matrix to respect the geometry of the probability manifold:
$$\Delta 	heta = -\eta \mathcal{G}^{-1}
abla_	heta F$$

### Stochastic Differential Equations (SDEs)
In continuous time, the flow of states can be described by:
$$dx = f(x)dt + \sqrt{2Q}dW$$
where $f(x)$ is the drift function derived from the free energy gradient.
