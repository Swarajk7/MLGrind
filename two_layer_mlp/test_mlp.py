import unittest
import numpy as np
from two_layer_mlp.mlp import Linear, ReLU, MLP2Layer

class TestMLP(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 2
        self.batch_size = 3
        
        self.x = np.random.randn(self.batch_size, self.input_dim)
        self.grad_output = np.random.randn(self.batch_size, self.output_dim)

    def test_linear_shapes(self):
        """Test if Linear layer produces correct output shapes."""
        layer = Linear(self.input_dim, self.hidden_dim)
        # Note: self.WDH and self.bH are initialized in mlp.py
        if not hasattr(layer, 'WDH') or layer.WDH is None: 
            self.skipTest("Weights not initialized in Linear layer")
            
        out = layer.forward(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.hidden_dim))

    def test_relu_shapes(self):
        """Test if ReLU layer produces correct output shapes."""
        layer = ReLU()
        out = layer.forward(self.x)
        if out is None:
            self.skipTest("ReLU forward not implemented")
        self.assertEqual(out.shape, self.x.shape)

    def test_mlp_shapes(self):
        """Test if MLP produces correct output shapes."""
        mlp = MLP2Layer(self.input_dim, self.hidden_dim, self.output_dim)
        out = mlp.forward(self.x)
        if out is None:
            self.skipTest("MLP forward not implemented")
        self.assertEqual(out.shape, (self.batch_size, self.output_dim))

    def test_gradient_check(self):
        """
        Numerically check gradients for a small MLP.
        This is the most critical test for backward pass.
        """
        mlp = MLP2Layer(self.input_dim, self.hidden_dim, self.output_dim)
        
        # We need a proper implementation to check gradients.
        # This test will likely fail or skip until the user implements the logic.
        xBD = np.random.randn(self.batch_size, self.input_dim)
        yBO = np.random.randn(self.batch_size, self.output_dim)
        
        yhatBO = mlp.forward(xBD)
        # using a MSE dummy loss to test the backprop implementation.
        dummy_loss = np.sum((yBO - yhatBO) ** 2)
        dLdyhatBO = 2 * (yhatBO - yBO)
        analytic_gradient = mlp.backward(dLdyhatBO)
        
        # Numerical check for gradient w.r.t input x
        # F returns the dummy loss MSE value.
        def f(x_val):
            y = mlp.forward(x_val)
            return np.sum((y-yBO) ** 2)
        
        numerical_gradient = numerical_gradient_check(f, xBD, epsilon=1e-7)
        self.assertTrue(np.allclose(numerical_gradient, analytic_gradient, rtol=1e-3))
        

        
def numerical_gradient_check(f, x, epsilon=1e-7):
    """
    Standard numerical gradient check implementation.
    f: function that takes x and returns a scalar loss
    x: numpy array
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + epsilon
        fx_plus = f(x)
        
        x[idx] = old_val - epsilon
        fx_minus = f(x)
        
        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        x[idx] = old_val
        it.iternext()
    return grad

if __name__ == '__main__':
    unittest.main()
