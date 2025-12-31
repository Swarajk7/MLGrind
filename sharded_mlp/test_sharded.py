import unittest
import numpy as np
from two_layer_mlp.mlp import MLP2Layer
from sharded_mlp.sharded_mlp import ShardedMLP2Layer

class TestShardedMLP(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 2
        self.batch_size = 3
        self.num_shards = 2
        
        # 1. Initialize Vanilla MLP
        self.vanilla_mlp = MLP2Layer(self.input_dim, self.hidden_dim, self.output_dim)
        
        # 2. Initialize Sharded MLP
        self.sharded_mlp = ShardedMLP2Layer(self.input_dim, self.hidden_dim, self.output_dim, self.num_shards)
        
        # 3. Manually shard the weights from vanilla to sharded for comparison
        self._shard_weights()

    def _shard_weights(self):
        """
        Helper to copy and shard weights from vanilla_mlp to sharded_mlp.
        This ensures we are comparing the exact same initialization.
        """
        v_l1 = self.vanilla_mlp.layers[0] # Linear 1
        v_l2 = self.vanilla_mlp.layers[2] # Linear 2
        
        for i in range(self.num_shards):
            s_l1 = self.sharded_mlp.shards[i]['linear1']
            s_l2 = self.sharded_mlp.shards[i]['linear2']
            
            # Column Sharding for Layer 1
            start_col = i * (self.hidden_dim // self.num_shards)
            end_col = (i + 1) * (self.hidden_dim // self.num_shards)
            s_l1.WDH_shard = v_l1.WDH[:, start_col:end_col].copy()
            s_l1.bH_shard = v_l1.bH[start_col:end_col].copy()
            
            # Row Sharding for Layer 2
            start_row = i * (self.hidden_dim // self.num_shards)
            end_row = (i + 1) * (self.hidden_dim // self.num_shards)
            s_l2.WDH_shard = v_l2.WDH[start_row:end_row, :].copy()
            s_l2.bH = v_l2.bH.copy() # Bias is usually same on all or handled once

    def test_forward_match(self):
        """Verify that sharded forward pass matches vanilla forward pass."""
        xBD = np.random.randn(self.batch_size, self.input_dim)
        
        vanilla_out = self.vanilla_mlp.forward(xBD)
        sharded_out = self.sharded_mlp.forward(xBD)
        
        if sharded_out is None:
            self.skipTest("Sharded forward not implemented")
            
        self.assertTrue(np.allclose(vanilla_out, sharded_out), "Forward pass output mismatch!")

    def test_backward_match(self):
        """Verify that sharded backward pass match vanilla backward pass gradients."""
        xBD = np.random.randn(self.batch_size, self.input_dim)
        dZBO = np.random.randn(self.batch_size, self.output_dim)
        
        # Run forward
        self.vanilla_mlp.forward(xBD)
        self.sharded_mlp.forward(xBD)
        
        # Run backward
        vanilla_grad_in = self.vanilla_mlp.backward(dZBO)
        sharded_grad_in = self.sharded_mlp.backward(dZBO)
        
        if sharded_grad_in is None:
            self.skipTest("Sharded backward not implemented")

        # 1. Check input gradients
        self.assertTrue(np.allclose(vanilla_grad_in, sharded_grad_in), "Input gradient mismatch!")
        
        # 2. Check weight gradients (accumulate shards)
        for i in range(self.num_shards):
            v_l1 = self.vanilla_mlp.layers[0]
            s_l1 = self.sharded_mlp.shards[i]['linear1']
            
            start_col = i * (self.hidden_dim // self.num_shards)
            end_col = (i + 1) * (self.hidden_dim // self.num_shards)
            
            # Compare dW shards
            self.assertTrue(np.allclose(v_l1.dWDH[:, start_col:end_col], s_l1.dWDH_shard), 
                            f"Layer 1 weight gradient mismatch in shard {i}")

if __name__ == '__main__':
    unittest.main()
