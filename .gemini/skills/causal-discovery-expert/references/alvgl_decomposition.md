# ALVGL Decomposition

Augmented Latent-Variable Graphical Lasso (ALVGL) is used to decompose a precision matrix (inverse covariance) into a sparse component representing direct causal links and a low-rank component representing latent confounders.

## Mathematical Formulation

$$ \Theta = S - L $$

Where:
- $\Theta$: The estimated precision matrix.
- $S$: Sparse matrix (direct causal structure).
- $L$: Low-rank matrix (latent variables).

## Optimization Objective

$$ \min_{S, L} \{ -\ln \det(S-L) + \langle S-L, \Sigma angle + \lambda \|S\|_1 + \gamma \mathrm{Tr}(L) \} $$

subject to: $S - L \succ 0$ and $L \succeq 0$.

## Neanderthal Implementation Strategy

1. **Precision Matrix ($\Sigma^{-1}$)**: Compute the sample covariance $\Sigma$ and its inverse.
2. **Proximal Operators**:
   - **Soft-thresholding** for the $L_1$ norm (Sparse $S$).
   - **Singular Value Thresholding (SVT)** for the nuclear norm (Low-rank $L$).
3. **Iteration**: Use ADMM to alternate between updating $S$ and $L$.

## Acyclicity Constraints

For Neural Causal Models, ensure the discovered structure is a Directed Acyclic Graph (DAG) using continuous constraints:
$$ h(W) = \mathrm{Tr}(\exp(W \circ W)) - d = 0 $$
where $W$ is the weighted adjacency matrix.
