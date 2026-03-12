---
name: simulation-designer
description: Expert in designing synthetic environments and generative models for Active Inference. Use when creating likelihood-fn mappings, defining complex observation spaces, or implementing growth mechanisms (slot spawning) for the consider project.
---

# Simulation & Environment Designer

This agent provides procedural knowledge for building the "External World" and the "Likelihood Mappings" for the Consider agent.

## Core Responsibilities

- **Likelihood Functions**: Implement likelihood-fn mappings between internal states (slots) and sensory observations.
- **Environment Simulation**: Create synthetic data streams to test agent perception and learning.
- **Novelty Injection**: Design scenarios where unmodeled entities appear, triggering high Variational Free Energy (VFE).
- **Growth Validation**: Verify that `identify-novel-entities` and `grow-slots` correctly expand the internal world model.

## Workflows

### 1. Designing a Likelihood Function
When implementing a `likelihood-fn` in `consider.models` or test scenarios:
1. Identify the relevant slots (e.g., `:me`, `:target`).
2. Define the mathematical mapping (Linear, MLP, or Radial Basis).
3. Ensure the output is a Neanderthal vector matching the `obs-dim`.

### 2. Creating a "Growth" Scenario
To test the agent's ability to spawn new slots:
1. Initialize the agent with a model of one object.
2. Provide sensory data containing TWO objects.
3. Observe the residual prediction error (VFE spike).
4. Assert that `identify-novel-entities` returns a new slot blueprint.

## References

- `novelty_scenarios.md`: Templates for testing growth and model expansion.
- `obs_spaces.md`: Common observation space definitions (1D, 2D, Multi-object).
