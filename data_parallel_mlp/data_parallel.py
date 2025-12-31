import numpy as np
from typing import List
from two_layer_mlp.mlp import MLP2Layer
import copy

def _sync_parameters(replicas: List[MLP2Layer], layer_idx: int, attr_name: str, average: bool = False) -> None:
    """
    Synchronizes gradients across all replicas using an All-Reduce operation.
    
    Note: In standard data parallel training, gradients are averaged across shards.
    However, to maintain parity with a single MLP instance during testing (where 
    loss scaling differs), we default to sum-reduction.
    """
    # Stack gradients from all replicas: (num_shards, ...)
    grads = np.array([getattr(r.layers[layer_idx], attr_name) for r in replicas])

    # All-Reduce Sum via einsum
    # If it's a 2D weight: 'sij->ij'. If it's a 1D bias: 'si->i'
    if grads.ndim == 3:
        total_grad = np.einsum('sij->ij', grads)
    else:
        total_grad = np.einsum('si->i', grads)
    
    if average:
        total_grad /= replicas

    # Broadcast back to all replicas
    for r in replicas:
        setattr(r.layers[layer_idx], attr_name, total_grad.copy())

class DataParallelMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_shards: int) -> None:
        """
        Data Parallelism replicates the model across multiple virtual devices.
        Weights are identical across shards initially.
        """
        self.num_shards = num_shards
        self.replicas: List[MLP2Layer] = []
        mlp2layer = MLP2Layer(input_dim, hidden_dim, output_dim)
        for i in range(num_shards):
            replica = copy.deepcopy(mlp2layer)
            self.replicas.append(replica)

    def forward(self, xBD: np.ndarray) -> np.ndarray:
        """
        The input batch xBD is split into num_shards mini-batches.
        Each replica computes forward pass on its sub-batch.
        Returns the concatenated output of share Batch by Output Size.
        """
        outputs = []
        batch_size_per_shard = xBD.shape[0] // self.num_shards
        for i in range(self.num_shards):
            xBD_shard = xBD[i * batch_size_per_shard : (i + 1) * batch_size_per_shard]
            outputs.append(self.replicas[i].forward(xBD_shard))
        return np.concatenate(outputs, axis=0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        1. Split upstream gradients across replicas.
        2. Run backward pass on each replica.
        3. Synchronize (AllReduce) gradients across all replicas by averaging them.
        Returns concatenated local input gradients.
        """
        batch_size_per_shard = grad_output.shape[0] // self.num_shards
        gradients = []
        # First do gradient computation in all replica, this would happen in parallel in GPU.
        for i in range(self.num_shards):
            grad_output_shard = grad_output[i * batch_size_per_shard : (i + 1) * batch_size_per_shard]
            gradients.append(self.replicas[i].backward(grad_output_shard))
        # Now do gradient synchronization.
        _sync_parameters(self.replicas, 0, 'dWDH')
        _sync_parameters(self.replicas, 0, 'dbH')
        _sync_parameters(self.replicas, 2, 'dWDH')
        _sync_parameters(self.replicas, 2, 'dbH')
        return np.concatenate(gradients, axis=0)

    def sync_parameters(self) -> None:
        """
        Synchronize weights and biases across all replicas.
        Typically called after an update to ensure no numerical drift.
        """
        _sync_parameters(self.replicas, 0, 'WDH', average=True)
        _sync_parameters(self.replicas, 0, 'bH', average=True)
        _sync_parameters(self.replicas, 2, 'WDH', average=True)
        _sync_parameters(self.replicas, 2, 'bH', average=True)
        
            
