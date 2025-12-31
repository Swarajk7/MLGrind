# MLGrind

A repository for learning machine learning from scratch.

*Developed with the assistance of Antigravity to test the codebase.*

## Project Structure

- `two_layer_mlp/`: Vanilla 2-layer MLP implementation and unit tests.
- `sharded_mlp/`: Virtual sharded MLP implementation (Model Parallelism) and comparison tests.
- `data_parallel_mlp/`: Data Parallel MLP implementation and comparison tests.

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

```bash
python3 -m unittest sharded_mlp/test_sharded.py
```

## Chapter 3: Data Parallelism

Implementation of Data Parallelism where the batch is split across identical model replicas.

### Architecture
- **Replication**: Model is replicated across $N$ virtual devices.
- **Batch Splitting**: Input batch is divided into sub-batches for each replica.
- **Synchronization**: Gradients are synchronized using an `AllReduce` operation after the backward pass.

```bash
python3 -m unittest data_parallel_mlp/test_data_parallel.py
```
