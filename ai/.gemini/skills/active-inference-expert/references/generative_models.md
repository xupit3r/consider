# Generative Models in Active Inference

The generative model is the heart of an Active Inference agent. it is a probabilistic representation of how sensory data is produced by hidden causes in the world.

## Formal Definition

A generative model is defined by the joint probability $P(y, x)$ of sensory observations ($y$) and hidden states ($x$):
$$P(y, x) = P(y|x)P(x)$$

### 1. Likelihood: $P(y|x)$
The likelihood represents the "sensory mapping"—the process by which hidden states generate observations.
- **Function**: Maps internal representations to expected sensory input.
- **Inference**: The agent minimizes the difference between predicted and actual sensory input.

### 2. Priors: $P(x)$
Priors represent the agent's beliefs about hidden states *before* receiving new data.
- **Constraints**: Guide the interpretation of ambiguous sensory data.
- **Empirical Priors**: In hierarchical models, higher-level outputs serve as priors for lower levels.
- **Preferences as Priors**: Goals and preferences (e.g., maintaining body temperature) are encoded as prior beliefs about expected states.

## The Role of the Generative Model

The generative model enables two primary processes:

### Perceptual Inference (Perception)
Updating internal beliefs (the posterior) to match sensory data.
- **Objective**: Minimize Variational Free Energy ($F$).
- **Outcome**: An accurate internal representation of the current world state.

### Active Inference (Action)
Changing the world or the agent's relationship to it to make sensory data match prior preferences.
- **Objective**: Minimize Expected Free Energy ($G$).
- **Outcome**: Goal-directed behavior and information-seeking (exploration).

## Model Structures

### Discrete Models (POMDPs)
Used for categorical states and observations (e.g., being in "Room A" vs. "Room B").
- **A-Matrix**: Likelihood (Mapping states to observations).
- **B-Matrix**: Transitions (How states change given actions).
- **C-Matrix**: Preferences (Prior beliefs about observations).
- **D-Matrix**: Initial state priors.

### Continuous Models
Used for physical variables (e.g., position, velocity, force).
- **Laplace Approximation**: Assuming Gaussian distributions for the recognition density.
- **Generalized Coordinates**: Representing states and their temporal derivatives (velocity, acceleration, etc.).

### Hierarchical Models
Models where higher levels represent more abstract, slower-changing states that provide context for lower, faster-changing levels.
- **Top-Down**: Expectations (Empirical Priors).
- **Bottom-Up**: Prediction Errors (Sensory Evidence).
