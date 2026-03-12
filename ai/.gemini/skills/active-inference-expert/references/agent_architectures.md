# Agent Architectures in Active Inference

Active Inference agents are structured as generative models that minimize free energy. This document categorizes different agent architectures and their components.

## Core Frameworks

### Active Inference Agent
The primary implementation supporting:
- **Belief Updating**: Updating internal states based on sensory input.
- **Planning (Policy Selection)**: Selecting sequences of actions (policies) that minimize expected free energy.
- **Learning**: Adapting the generative model parameters over time.

### Generic POMDP Framework
A flexible framework for Partially Observable Markov Decision Processes, suitable for discrete state spaces.
- **State Space (S)**: Hidden states of the world.
- **Observation Space (O)**: Sensory inputs.
- **Action Space (A)**: Possible actions.
- **Transition Matrix (B)**: How states change given actions.
- **Observation Matrix (A)**: How states generate observations.

## Specialized Implementations

### Ant Colony (Stigmergy)
Swarm intelligence models where agents interact via the environment (e.g., pheromone trails).
- **Stigmergy**: Action is guided by environmental changes made by other agents.
- **Collective Active Inference**: Multiple agents minimizing a shared free energy objective.

### BioFirm
Biological firm theory applying active inference to organizational structures.
- **Organizational Markov Blankets**: Defining the boundary of a firm.
- **Hierarchical Active Inference**: Scaling from individuals to departments to the whole organization.

### Continuous Time Agents
Agents operating in continuous state-spaces, often using Stochastic Differential Equations (SDEs) and Laplace approximations.
- **Generalized Coordinates of Motion**: Representing states, velocities, accelerations, etc.

## Agent Components

### 1. Generative Model
The internal representation of the world's causal structure.
- **Prior Beliefs**: Initial assumptions about the world.
- **Likelihood**: Relationship between hidden states and observations.

### 2. Inference Engine
The mechanism for updating beliefs.
- **Variational Bayes**: Approximating the posterior distribution.
- **Message Passing**: Efficient belief propagation in graphical models.

### 3. Policy Selection (Expected Free Energy)
Choosing actions by predicting their consequences.
- **Epistemic Value (Information Gain)**: Actions that resolve uncertainty.
- **Pragmatic Value (Goal Realization)**: Actions that lead to preferred states.

## Development Tools

- **code/tools/src/models/active_inference/**: Core logic for agent implementation.
- **code/Things/**: Directory for specific agent instances (e.g., Ant_Colony, Simple_POMDP).
- **knowledge_base/agents/**: Theoretical foundations and design patterns.
