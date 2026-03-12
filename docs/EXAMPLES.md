# Consider Agent: Example Use Cases

This document describes practical applications of the **Consider** Active Inference agent, illustrating how it handles perception, planning, and learning.

## 1. Adaptive Climate Control (Smart Thermostat)
**Objective**: Maintain a target temperature.
**Active Inference Mechanism**: Pragmatic Value (Risk).
The agent compares its current belief about the room temperature with its preferences ($C$-matrix). It chooses actions (Heating/Cooling) that minimize the expected divergence from the target temperature.
- **State**: `[Temperature, HeaterState]`
- **Preference**: `[22.0, 0.0]`
- **Action**: `TURN_ON_HEATER` (chosen when $T < 22.0$)

## 2. Novelty Discovery (Dynamic Environment)
**Objective**: Adapt to unexpected changes in the environment.
**Active Inference Mechanism**: Generative Model Growth.
When the agent receives sensory data that cannot be explained by its current hidden states (slots), it identifies a "prediction error surplus." It then dynamically spawns a new slot and expands its recognition neural network to incorporate the new entity.
- **Scenario**: A new object appears in the field of view.
- **Result**: Agent grows from 1 slot to 2 slots automatically.

## 3. Social Navigation (Collaborative Assistant)
**Objective**: Assist a user while minimizing frustration.
**Active Inference Mechanism**: LLM-based State Scoring.
The agent uses a Large Language Model (LLM) as a "scent provider" during MCTS reasoning. The LLM predicts the likely social consequences of actions (e.g., explaining a concept vs. asking for feedback) and scores them based on progress and user satisfaction.
- **Hidden States**: `[UserProgress, UserFrustration]`
- **Reasoning**: MCTS tree search guided by LLM EFE estimates.

## 4. Epistemic Exploration (The Curious Robot)
**Objective**: Resolve uncertainty to enable future goal-directed action.
**Active Inference Mechanism**: Epistemic Value (Ambiguity).
In situations where the goal is obscured (e.g., a dark room), the agent prioritizes actions that reduce the ambiguity of its observation model, even if those actions don't provide immediate pragmatic progress.
- **Scenario**: Turning on a light to see the goal.
- **Result**: Agent chooses `TURN_ON_LIGHT` because it has high epistemic value (minimizes G), even if it doesn't move it closer to the exit.

---

### Running the Examples
You can run the executable versions of these examples using the test runner:
```bash
clojure -M:test -n consider.examples-test
```
