import unittest
import numpy as np
from two_layer_mlp.mlp import MLP2Layer
from data_parallel_mlp.data_parallel import DataParallelMLP

class TestDataParallel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 2
        self.batch_size = 4 # Must be divisible by num_shards
        self.num_shards = 2
        
        # 1. Initialize Vanilla MLP
        self.vanilla_mlp = MLP2Layer(self.input_dim, self.hidden_dim, self.output_dim)
        
        # 2. Initialize Data Parallel MLP
        self.dp_mlp = DataParallelMLP(self.input_dim, self.hidden_dim, self.output_dim, self.num_shards)
        
        # 3. Synchronize initial weights from vanilla to DP replicas
        self._sync_weights()

    def _sync_weights(self):
        """Helper to copy weights from vanilla_mlp to all DP replicas."""
        for i in range(self.num_shards):
            replica = self.dp_mlp.replicas[i]
            # Layer 1
            replica.layers[0].WDH = self.vanilla_mlp.layers[0].WDH.copy()
            replica.layers[0].bH = self.vanilla_mlp.layers[0].bH.copy()
            # Layer 2
            replica.layers[2].WDH = self.vanilla_mlp.layers[2].WDH.copy()
            replica.layers[2].bH = self.vanilla_mlp.layers[2].bH.copy()

    def test_forward_match(self):
        """Verify DP forward pass matches vanilla forward pass."""
        x = np.random.randn(self.batch_size, self.input_dim)
        
        vanilla_out = self.vanilla_mlp.forward(x)
        dp_out = self.dp_mlp.forward(x)
        
        if dp_out is None:
            self.skipTest("Data Parallel forward not implemented")
            
        self.assertTrue(np.allclose(vanilla_out, dp_out), "Forward pass output mismatch!")

    def test_backward_match(self):
        """Verify DP backward pass gradients match vanilla gradients averaged over replicas."""
        x = np.random.randn(self.batch_size, self.input_dim)
        grad_output = np.random.randn(self.batch_size, self.output_dim)
        
        # Forward
        self.vanilla_mlp.forward(x)
        self.dp_mlp.forward(x)
        
        # Backward
        vanilla_dx = self.vanilla_mlp.backward(grad_output)
        dp_dx = self.dp_mlp.backward(grad_output)
        
        if dp_dx is None:
            self.skipTest("Data Parallel backward not implemented")

        # 1. Check input gradients
        self.assertTrue(np.allclose(vanilla_dx, dp_dx), "Input gradient mismatch!")
        
        # 2. Check parameter gradients (vanilla should match averaged DP gradients)
        for i in range(self.num_shards):
            replica = self.dp_mlp.replicas[i]
            # Note: After DP backward, replica.layers[i].dWDH should be the synchronised (averaged) gradient
            self.assertTrue(np.allclose(self.vanilla_mlp.layers[0].dWDH, replica.layers[0].dWDH, rtol=1e-5), 
                            f"Layer 1 gradient mismatch in shard {i}")
            self.assertTrue(np.allclose(self.vanilla_mlp.layers[2].dWDH, replica.layers[2].dWDH, rtol=1e-5), 
                            f"Layer 2 gradient mismatch in shard {i}")

if __name__ == '__main__':
    unittest.main()
