# MCTS and Forest-of-Thought Patterns

System 2 reasoning involves exploring potential futures (trajectories of thought or action) to find the one that best satisfies an objective (e.g., minimizing Expected Free Energy).

## 1. Monte Carlo Tree Search (MCTS) with Active Inference

Standard MCTS uses UCT (Upper Confidence Bound for Trees):
$$ UCT(s, a) = Q(s, a) + C \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} $$

In Active Inference:
- **Value (Q)**: $-G(\pi)$, where $G$ is the Expected Free Energy.
- **Policy Prior (P)**: Output of the LLM (System 1).
- **Visits (N)**: Number of times a reasoning branch has been explored.

## 2. Forest-of-Thought (FoT)

FoT manages multiple, potentially disconnected, reasoning trees simultaneously. This allows the agent to switch between different "lines of reasoning" as new evidence arrives.

- **Dynamic Branching**: Expanding the tree based on the probability of a reasoning step.
- **Root Switching**: Changing the focus to a different tree in the forest if its EFE is lower.

## 3. Recursive Critic

A mechanism for a node to "criticize" its parent or children, updating the $Q$ value based on consistency and logical coherence.

- **Self-Correction**: A child node might find a contradiction in the parent's state, leading to a high "penalty" value that propagates back up the tree.

## 4. Policy Selection Workflow

1. **Selection**: Traverse the tree from the root using UCT until a leaf node is reached.
2. **Expansion**: Use the LLM to generate candidate steps from the leaf.
3. **Simulation/Evaluation**: Estimate the EFE of the new steps.
4. **Backpropagation**: Update the $Q$ and $N$ values of all nodes in the path back to the root.
