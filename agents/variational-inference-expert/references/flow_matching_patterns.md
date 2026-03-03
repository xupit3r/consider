# Flow Matching and DDVI Patterns

Amortized variational inference is used to map complex posterior distributions (beliefs about hidden states) to simple, tractable priors (e.g., standard normal) using neural networks and differential equations.

## 1. Flow Matching (FlowNP)

Flow matching defines a continuous-time flow that transforms a simple noise distribution $p_0$ into the target data distribution $p_1$ (the posterior $q(s|o)$).

- **Vector Field ($v_t$)**: A neural network that predicts the velocity of points at time $t$.
- **ODE Solver**: Used to integrate the vector field from $t=0$ to $t=1$.
  - **Euler Method**: $x_{t+dt} = x_t + v_t(x_t, t)dt$
  - **Runge-Kutta (RK4)**: More accurate higher-order solver.
- **Conditional Flow**: The vector field depends on context $c$ (sensory observations).

## 2. Diffusion-Based Variational Inference (DDVI)

DDVI uses a diffusion process as the variational family, aligning the generative model and the recognition model through a specialized objective function.

- **Wake-Sleep ELBO**:
  - **Wake Phase**: Update the recognition model $q(s|o)$ using actual sensory data.
  - **Sleep Phase**: Update the generative model $p(o,s)$ using samples from the recognition model.
- **Score Matching**: Learning the gradient of the log-density ($
abla \ln p(x)$).

## 3. Amortized Inference Workflow

1. **Context Encoding**: Use a permutation-invariant encoder (e.g., Deep Set or Transformer) to process sensory context points.
2. **Flow Prediction**: The encoder's output parameterizes the vector field $v_t$.
3. **Sampling**: 
   - Draw a sample $x_0$ from the standard normal prior.
   - Solve the ODE $dx/dt = v_t(x, t, c)$ from $t=0$ to $t=1$.
   - The result $x_1$ is a sample from the posterior $q(s|o, c)$.

## 4. Variational Metrics

- **ELBO**: Evidence Lower Bound ($F$).
- **Complexity**: $KL(q(s) \| p(s))$ (Ensuring the posterior doesn't deviate too much from the prior).
- **Accuracy**: $E_q[\ln p(o|s)]$ (Ensuring the posterior explains the observations).
