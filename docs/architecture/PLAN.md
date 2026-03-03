# Consider Inference Engine: Implementation Plan (Active Inference Refined)

## Overview
"Consider" is a Unified Causal System 2 Agent built in Clojure. It implements an **Active Inference** agent that maintains an internal world model, updates its beliefs via amortized variational inference, and selects reasoning policies that minimize **Expected Free Energy (G)**.

## Subsystem Architecture & Responsibilities

### 1. Executive Orchestrator (Policy Selection)
**Role:** The System 2 reasoning core responsible for choosing the optimal reasoning policy (π).
**Implementation Tasks:**
*   Implement the **Forest-of-Thought (FoT)** to explore potential futures (trajectories of belief states).
*   Utilize **MCTS** where the node value (Q) is defined by the **Expected Free Energy (G)**.
*   Decompose G into **Risk** (pragmatic value/utility) and **Ambiguity** (epistemic value/information gain).
*   Use the LLM (System 1) to generate the **Policy Prior (P)** for UCT expansion.
*   Implement **Sparse Activation** to prune branches with high expected free energy.

### 2. Probabilistic World Model (Generative Model)
**Role:** Maintenance of the agent's internal states (μ) and their relationship to sensory data (o).
**Implementation Tasks:**
*   Implement the **Likelihood Mapping P(o|s)** and the transition dynamics between hidden states.
*   Refine the **Object-Centric Slots** to represent hidden variables ($s$) in the generative model.
*   Define agent **Preferences (C-matrix)** as prior beliefs over sensory states (goals).
*   Implement "grow-and-merge" logic for dynamic state-space expansion (adding/merging slots).

### 3. Causal Discovery Engine (Structure Learning)
**Role:** Discovering the causal dependencies between hidden states (Internal States).
**Implementation Tasks:**
*   Perform ALVGL decomposition using `neanderthal` to find direct causal links (sparse) and latent confounders (low-rank).
*   Enforce continuous acyclicity to ensure the generative model is a valid DAG.
*   Update the transition matrices (B) of the world model based on discovered causal links.

### 4. Density/Sampling Subsystem (Perceptual Inference)
**Role:** Updating the **Recognition Density q(s)** to minimize **Variational Free Energy (F)**.
**Implementation Tasks:**
*   Implement **Flow Matching** (FlowNP) to flow from a simple prior to a complex posterior belief state.
*   Utilize **DDVI** with a wake-sleep ELBO objective to align the recognition model with the generative model.
*   Decompose F into **Complexity** (KL divergence from priors) and **Accuracy** (expected log likelihood).
*   Implement ODE solvers for continuous-time belief updating.

### 5. LLM Abstraction Layer (System 1)
**Role:** Providing fast, heuristic priors for the executive core.
**Implementation Tasks:**
*   Implement the `PolicyPredictor` to provide the prior $P(π)$.
*   Implement the `ProcessScorer` to provide initial heuristic estimates of pragmatic and epistemic value.

## Agent Handoff Protocol
This specification directory serves as the rigid contract between subagents.
1. When implementing a subsystem, an agent MUST depend exclusively on the specs defined here to ensure interoperability.
2. Changes to these `.clj` spec files require a consensus review, as they represent the architectural interfaces of the Unified Agent.
