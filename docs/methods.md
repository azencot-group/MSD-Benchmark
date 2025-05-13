# MSD Benchmark: Method Integration

The MSD (Multi-factor Sequential Disentanglement) benchmark supports a modular and configurable approach to evaluating representation learning methods. Adding a new method to the benchmark involves implementing a model and a training workflow that conform to a common API.

This document outlines:
- The structure of the `AbstractModel` and `AbstractTrainer`
- Requirements for new methods

For configuration and initialization details, see [docs/configuration.md](docs/configuration.md).

---

## AbstractModel Interface

All models in MSD must inherit from `AbstractModel`, which itself extends `torch.nn.Module`. It defines a consistent API for encoding, decoding, sampling, and manipulating latent spaces.

### Required Methods
- `encode(x: Tensor) -> Tensor`: Encode raw input to latent space.
- `decode(z: Tensor) -> Tensor`: Reconstruct input from latent space.
- `latent_dim() -> int`: Report the dimensionality of the latent space.
- `latent_vector(x: Tensor) -> Tensor`: Extract latent representations as a single vector.
- `sample(z: int) -> Tensor`: Generate synthetic samples from latent codes.
- `swap_channels(z1, z2, C) -> Tensor`: Perform channel-level factor swapping.
- `forward(x: Tensor) -> Tensor`: Define the forward pass.

### Optional Hooks
- `preprocess(x: Tensor) -> Tensor`: Input preprocessing (e.g., normalization).
- `postprocess(x: Tensor) -> Tensor`: Output transformation.

---

## AbstractTrainer Interface

The `AbstractTrainer` class defines the training lifecycle and is responsible for:
- Initializing models, optimizers, and evaluators
- Handling checkpointing and logging
- Executing training and evaluation loops

### Required Method
- `train_step(epoch: int) -> dict`: A single epoch of model updates.

### Lifecycle
1. Initialize model and data
2. For each epoch:
   - Run `train_step`
   - Log metrics and save state
   - Evaluate model periodically and store best version

The class handles resuming from checkpoints and saving relevant metadata (`classes`, `config`, etc.)

---

## Summary

The MSD benchmark separates modeling, training, and evaluation into pluggable modules. By adhering to the `AbstractModel` and `AbstractTrainer` interfaces, you can integrate new learning methods with minimal friction.

To see a working example, check the included methods directory. For experiment setup and configuration, refer to [docs/configuration.md](docs/configuration.md).
