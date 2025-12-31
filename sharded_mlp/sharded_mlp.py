import numpy as np

"""
Notation:
W: Weight matrix
D: Input dimension
H: Hidden dimension
O: Output dimension
B: Batch size
"""

class ShardedLinearColumn:
    def __init__(self, in_features, out_features, num_shards, shard_id):
        """
        Matrix W is split into columns.
        W_shard shape: (in_features, out_features // num_shards)
        """
        self.num_shards = num_shards
        self.shard_id = shard_id

        assert out_features % num_shards == 0, "out_features must be divisible by num_shards"
        
        # TODO: Initialize sharded weights and biases
        self.WDH_shard = np.random.randn(in_features, out_features // num_shards) * np.sqrt(2 / in_features)
        self.bH_shard = np.zeros(out_features // num_shards)

        self.dWDH_shard = None
        self.dbH_shard = None
        self.xBD = None

    def forward(self, xBD):
        """
        Each shard computes a piece of the output dimension.
        Output shape: (B, out_features // num_shards)
        """
        self.xBD = xBD
        return np.einsum('BD, Dh -> Bh', xBD, self.WDH_shard) + self.bH_shard

    def backward(self, dZBH_shard):
        """
        dZBH_shard shape: (B, out_features // num_shards)
        Returns dXBD (B, in_features)
        """
        self.dWDH_shard = np.einsum('BD, Bh -> Dh', self.xBD, dZBH_shard)
        self.dbH_shard = np.einsum('Bh -> h', dZBH_shard)
        return np.einsum('Bh, Dh -> BD', dZBH_shard, self.WDH_shard)

class ShardedLinearRow:
    def __init__(self, in_features, out_features, num_shards, shard_id):
        """
        Matrix W is split into rows.
        W_shard shape: (in_features // num_shards, out_features)
        """
        self.num_shards = num_shards
        self.shard_id = shard_id
        assert in_features % num_shards == 0, "in_features must be divisible by num_shards"
        # TODO: Initialize sharded weights and biases
        self.WDH_shard = np.random.randn(in_features // num_shards, out_features) * np.sqrt(2 / in_features) 
        self.bH = np.zeros(out_features) 
        self.x_shard = None

        self.dWDH_shard = None
        self.dbH = None

    def forward(self, x_shard):
        """
        Each shard computes a partial sum of the output.
        x_shard shape: (B, in_features // num_shards)
        Output shape: (B, out_features)
        """
        self.x_shard = x_shard
        # W is now 
        # [ W1
        #   W2 ]
        return np.einsum('Bd, dH -> BH', x_shard, self.WDH_shard) + self.bH

    def backward(self, dZBH):
        """
        dZBH shape: (B, out_features)
        Returns dX_shard (B, in_features // num_shards)
        """
        self.dWDH_shard = np.einsum('Bd, BH -> dH', self.x_shard, dZBH)
        self.dbH = np.einsum('BH -> H', dZBH)
        return np.einsum('BH, dH -> Bd', dZBH, self.WDH_shard)

class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, xBD):
        self.mask = (xBD > 0).astype(float)
        return np.maximum(0, xBD)
    
    def backward(self, dZBD):
        return np.where(self.mask, dZBD, 0)

class ShardedMLP2Layer:
    def __init__(self, input_dim, hidden_dim, output_dim, num_shards):
        self.num_shards = num_shards
        
        # Each "device" (shard_id) has its own local layers
        self.shards = []
        for i in range(num_shards):
            shard_layers = {
                'linear1': ShardedLinearColumn(input_dim, hidden_dim, num_shards, i),
                'relu': ReLU(), # ReLU is element-wise, can be simulated locally
                'linear2': ShardedLinearRow(hidden_dim, output_dim, num_shards, i)
            }
            self.shards.append(shard_layers)

    def forward(self, xBD):
        """
        Simulate the parallel forward pass across all shards.
        1. Layer 1 (Column): Each shard gets full X, produces partial Hidden.
        2. ReLU: Local to each shard.
        3. Layer 2 (Row): Each shard gets partial Hidden, produces partial Output.
        4. AllReduce: Sum partial Outputs.
        """
        # TODO: Implement simulated parallel forward pass
        outputs = []
        for i in range(self.num_shards):
            linear1_out_Bh = self.shards[i]['linear1'].forward(xBD)
            relu_out_Bh = self.shards[i]['relu'].forward(linear1_out_Bh)
            linear2_out_BO = self.shards[i]['linear2'].forward(relu_out_Bh)
            outputs.append(linear2_out_BO)
        # simulate all reduce operation
        return sum(outputs)
        
            

    def backward(self, dZBO):
        """
        Simulate the parallel backward pass.
        1. Layer 2 (Row): Input dZBO, Output partial dHidden.
        2. ReLU: Local.
        3. Layer 1 (Column): Input partial dHidden, Output partial dX.
        4. AllReduce: Sum partial dX (though usually input gradients aren't summed in MP)
        """
        # TODO: Implement simulated parallel backward pass
        gradients = []
        for i in range(self.num_shards):
            linear2_grad_Bh = self.shards[i]['linear2'].backward(dZBO)
            relu_grad_Bh = self.shards[i]['relu'].backward(linear2_grad_Bh)
            linear1_grad_BD = self.shards[i]['linear1'].backward(relu_grad_Bh)
            gradients.append(linear1_grad_BD)
        # simulate all reduce operation
        return sum(gradients)
