# MLGrind

A repository for learning machine learning from scratch.

## Project Structure

- `two_layer_mlp/`: Vanilla 2-layer MLP implementation and unit tests.
- `sharded_mlp/`: Virtual sharded MLP implementation (Model Parallelism) and comparison tests.

## Chapter 1: 2-Layer MLP

Implementation of a basic neural network with one hidden layer using only NumPy.

### Key Components
- **Linear Layer**: Computes $Y = XW + b$.
- **ReLU**: Implements $f(x) = \max(0, x)$.
- **Testing**: Includes **Numerical Gradient Checking**.

```bash
python3 -m unittest two_layer_mlp/test_mlp.py
```

## Chapter 2: Virtual Sharding

Extension of the 2-layer MLP to support "virtual sharding" (Model Parallelism).

### Architecture (Megatron-style)
- **Column Parallelism**: First linear layer is split across output dimensions.
- **Row Parallelism**: Second linear layer is split across input dimensions.
- **Synchronization**: Simulated `AllReduce` (sum) across virtual shards.

### Testing
Verification compares sharded outputs and gradients against the vanilla implementation.

```bash
python3 -m unittest sharded_mlp/test_sharded.py
```
