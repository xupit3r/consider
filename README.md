# Consider

**Consider** is a high-performance implementation of a **Unified Reasoning Loop** for an **Active Inference** agent. It leverages the Free Energy Principle (FEP) to create an autonomous agent capable of perception, inference, learning, decision-making, and action within dynamic environments.

## Project Overview

The project is built in **Clojure**, utilizing state-of-the-art numerical computing and deep learning libraries to ensure mathematical rigor and execution efficiency.

### Core Technologies

- **Language**: Clojure (1.12.4)
- **Numerical Computing**: [Neanderthal](https://neanderthal.uncomplicate.org/) (BLAS/LAPACK) for high-performance matrix and vector operations.
- **Deep Learning**: [Deep Diamond](https://deep-diamond.uncomplicate.org/) for neural network parameterization.
- **Data Manipulation**: `tech.ml.dataset`
- **Asynchronous Programming**: `promesa`

## Architecture

The agent operates through an iterative cycle: **Perceive -> Infer -> Learn -> Decide -> Act -> Sleep**.

- **`consider.core`**: Orchestrates the global reasoning loop and component interactions.
- **`consider.world-model`**: Manages the generative model, hidden states ("slots"), and transition dynamics. Supports dynamic slot growth for novel entity discovery.
- **`consider.inference`**: Performs belief updates by minimizing **Variational Free Energy (VFE)**. Utilizes **Flow Matching** for continuous state estimation.
- **`consider.causal`**: Implements causal structure discovery (**DAG learning**) from belief trajectories using the **ALVGL** (Sparse-Low Rank) decomposition with NOTEARS acyclicity constraints.
- **`consider.executive`**: Handles policy selection using **Monte Carlo Tree Search (MCTS)**, minimizing **Expected Free Energy (EFE)**. Incorporates causal epistemic guidance.
- **`consider.models`**: Defines high-performance neural models (MLPs) using Neanderthal for amortized inference.
- **`consider.llm`**: Integrates LLMs (Mock, Ollama, and Dynamic providers) to serve as System 1 predictors and scorers for reasoning trees.
- **`consider.specs`**: Formal data structure specifications using `clojure.spec.alpha`.

## Key Features

- **High-Performance Inference**: Refactored core mathematical operations to use native BLAS/LAPACK via Neanderthal.
- **Amortized Recognition**: Implementation of a **Sleep Phase** training loop that optimizes a neural vector field to match generative flows, enabling fast amortized inference.
- **Causal Reasoning**: Robust structure learning that identifies dependencies between hidden states, allowing the agent to refine its world model over time.
- **Structured Reasoning**: MCTS-based decision making that balances exploration (epistemic value) and exploitation (pragmatic utility).

## Building and Running

### Prerequisites

- [Clojure CLI](https://clojure.org/guides/install_clojure)
- Native BLAS/LAPACK libraries (e.g., Intel MKL or OpenBLAS).
- [Ollama](https://ollama.ai/) (optional, for real LLM support).

### Key Commands

- **Run Tests**: `clojure -M:test`
- **Start nREPL**: `clojure -M:test:nrepl` (Starts a server on port 7888 with test paths included).
- **Interactive Development**: Use the nREPL server to evaluate code and inspect agent states in real-time.

## Project Organization

- `src/`: Source code.
- `test/`: Unit and integration tests.
- `ai/`: Central directory for AI-related configuration, including Gemini skills, agents, and MCP settings.
- `research/`: Mathematical and architectural background documentation.
- `docs/`: Usage examples and detailed design plans.

## License

Copyright © 2026

Distributed under the Apache License 2.0.
