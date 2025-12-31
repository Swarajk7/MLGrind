# MLGrind

A repository for learning machine learning from scratch.

## Project Structure

- `two_layer_mlp/`:
    - `mlp.py`: Implementation of a 2-layer Multi-Layer Perceptron using only NumPy.
    - `test_mlp.py`: Unit tests for verifying forward/backward pass and gradient checking.

## Exercise: 2-Layer MLP

The goal of this exercise is to implement a basic neural network with one hidden layer using only NumPy and basic linear algebra.

### Key Components

1. **Linear Layer (`Linear`)**: 
   - Performs $Y = XW + b$ where $W$ is the weight matrix and $b$ is the bias vector.
   - Implements both forward and backward passes.
2. **ReLU Activation (`ReLU`)**:
   - Implements the Rectified Linear Unit activation function: $f(x) = \max(0, x)$.
   - Handles the gradient computation for the backward pass.
3. **MLP Model (`MLP2Layer`)**:
   - Composes two linear layers and a ReLU activation in the sequence: `Linear` -> `ReLU` -> `Linear`.
   - Manages the forward and backward flow of data through the network.

### Implementation Details

- **Variable Naming**: Uses custom notation (e.g., `WDH` for Weight Matrix with dimensions Input-Depth x Hidden-Depth).
- **Initialization**: Weights are initialized using a small random distribution (standard normal * 0.01) to break symmetry.
- **Backpropagation**: Gradients are calculated using the chain rule and stored in each layer (`dWDH`, `dbH`).
- **Matrix Operations**: Uses `np.einsum` for efficient and readable tensor contractions.

### Testing

Verification is performed via unit tests, including **Numerical Gradient Checking** to ensure the analytic gradients match numerical derivatives.

To run the tests:
```bash
python3 -m unittest two_layer_mlp/test_mlp.py
```
