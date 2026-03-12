# Consider Project Overview

The **Consider** project is an implementation of a Unified Reasoning Loop for an Active Inference agent. It leverages the Free Energy Principle (FEP) to create an autonomous agent that can perceive, infer, learn, decide, and act within its environment. The project is built in Clojure, utilizing high-performance numerical computing libraries for its core mathematical operations.

## Core Technologies

- **Language**: Clojure (1.12.4)
- **Numerical Computing**: [Neanderthal](https://neanderthal.uncomplicate.org/) (BLAS/LAPACK)
- **Deep Learning/Tensors**: [Deep Diamond](https://deep-diamond.uncomplicate.org/)
- **Data Manipulation**: `tech.ml.dataset`
- **Asynchronous Programming**: `promesa`

## Architecture

The agent operates through an iterative cycle (Perceive -> Infer -> Learn -> Decide -> Act -> Sleep), with the following core modules:

- `consider.core`: Orchestrates the global reasoning loop and handles the interaction between components.
- `consider.world-model`: Manages the generative model, hidden states (slots), and transition dynamics.
- `consider.inference`: Performs belief updates by minimizing Variational Free Energy (VFE). Includes training of recognition models.
- `consider.causal`: Implements causal structure discovery (DAG learning) from belief trajectories.
- `consider.executive`: Handles policy selection and MCTS reasoning by minimizing Expected Free Energy (EFE).
- `consider.llm`: Integrates LLMs (Mock and Ollama supported) to serve as predictors and scorers for reasoning trees.
- `consider.specs.*`: Formal data structure specifications using `clojure.spec.alpha`.

## Building and Running

### Prerequisites
- [Clojure CLI](https://clojure.org/guides/install_clojure)
- Native BLAS/LAPACK libraries (e.g., Intel MKL or OpenBLAS) for Neanderthal.
- [Ollama](https://ollama.ai/) (for real LLM support, recommended model: `llama3`)

### Key Commands
- **Run Tests**: `clojure -M:test` (Uses `cognitect.test-runner`)
- **Start nREPL**: `clojure -M:nrepl` (Starts an nREPL server on port 7888)
- **Interactive Development**: Use the nREPL server to evaluate Clojure code and inspect agent states in real-time.

## Development Conventions

- **Mathematical Rigor**: Implementations must align with the mathematical foundations of Active Inference (VFE, EFE, Markov Blankets).
- **Clojure Idioms**: Prioritize functional patterns and immutable data structures where possible, while using Neanderthal's mutable vectors/matrices for performance-critical sections.
- **Specification Driven**: Use the `specs/` directory to define and validate data structures.
- **Performance**: Leverage Neanderthal and Deep Diamond for all tensor and matrix operations.
- **Testing**: Every new feature or bug fix must be accompanied by a test in the `test/` directory. Use mock LLMs for testing reasoning logic.

## Example Use Cases

A set of illustrative use cases showing the agent in action can be found in [docs/EXAMPLES.md](docs/EXAMPLES.md). These include:
- **Smart Thermostat**: Preference matching for climate control.
- **Novelty Discovery**: Dynamic generative model expansion.
- **Social Assistant**: LLM-guided state scoring for interaction.
- **Curious Explorer**: Information-seeking (Epistemic Value) under uncertainty.

To run the executable examples:
```bash
clojure -M:test -n consider.examples-test
```

## Specialized Skills

The project includes several specialized Gemini skills located in the `agents/` directory:
- `active-inference-expert`: Core active inference principles and generative models.
- `causal-discovery-expert`: ALVGL decomposition and DAG constraints.
- `neanderthal-expert`: High-performance numerical computing and memory management.
- `reasoning-architecture-expert`: MCTS and Forest-of-Thought (FoT) patterns.
- `variational-inference-expert`: Flow matching and DDVI.

Use `activate_skill <skill-name>` to load the specific knowledge and workflows for these domains.
