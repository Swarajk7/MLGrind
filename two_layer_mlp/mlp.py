import numpy as np

"""
D = Input feature Depth
H = Hidden feature Depth
B = Batch Size 
"""

class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize weights and biases.
        Weights should be initialized using a small random distribution (e.g., standard normal * 0.01).
        Biases should be initialized to zeros.
        """
        # He / Kaiming Initialization
        self.WDH: np.ndarray = np.random.randn(in_features, out_features) * np.sqrt(2/in_features)
        self.bH: np.ndarray = np.zeros(out_features)
        self.dWDH: np.ndarray | None = None
        self.dbH: np.ndarray | None = None
        self.xBD: np.ndarray | None = None # Cache input for backward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass: y = x @ W + b
        """
        self.xBD = x
        yBH = np.einsum('BD, DH -> BH', x, self.WDH) + self.bH
        return yBH

    def backward(self, dZBH: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass:
        - dL/dx: Gradient with respect to input
        - dL/dW: Gradient with respect to weights
        - dL/db: Gradient with respect to biases

        yBH = xBD @ WDH + bH
        L = f(yBH, label)
        dL/dWBH = xBD.T @ dZBH
        dL/dbH = np.sum(dZBH, axis=0)
        """
        self.dWDH = np.einsum('BD, BH -> DH', self.xBD, dZBH)
        self.dbH = np.einsum('BH -> H', dZBH)
        return np.einsum('BH, DH -> BD', dZBH, self.WDH)

class ReLU:
    def __init__(self) -> None:
        self.mask: np.ndarray | None = None # Cache mask for backward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass: y = max(0, x)
        """
        yBH = np.maximum(0, x)
        self.mask = (x > 0).astype(int)
        return yBH

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass (gradient of ReLU).
        """
        return grad_output * self.mask

class MLP2Layer:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Initialize a 2-layer MLP: Linear -> ReLU -> Linear
        """
        self.layers = [
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass through all layers in reverse order.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output