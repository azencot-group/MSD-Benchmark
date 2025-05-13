# MSD Benchmark: Method Integration

The MSD (Multi-factor Sequential Disentanglement) benchmark supports a modular and configurable approach to evaluating representation learning methods. Adding a new method to the benchmark involves implementing a model and a training workflow that conform to a common API.

This document outlines:
- The structure of the `AbstractModel` and `AbstractTrainer`
- Requirements for new methods

For configuration and initialization details, see [docs/configuration.md](configuration.md).

---

## AbstractModel Interface

All models in MSD must inherit from `AbstractModel`, which itself extends `torch.nn.Module`. It defines a consistent API for encoding, decoding, sampling, and manipulating latent spaces.

### Required Methods
- `encode(x: Tensor) -> Tensor`: Encode raw input to latent space (does not require a flat vector representation).
- `decode(z: Tensor) -> Tensor`: Reconstruct input from latent space.
- `latent_dim() -> int`: Report the dimensionality of the latent space.
- `latent_vector(x: Tensor) -> Tensor`: Extract latent representations as a single vector (flat vector).
- `sample(z: Tensor) -> Tensor`: Generate synthetic samples from latent codes.
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
- Optional: The constructor may set up additional training-specific components like loss functions.

### Lifecycle
1. Initialize model and data
2. For each epoch:
   - Run `train_step`
   - Log metrics and save state
   - Evaluate model periodically and store best version

The class handles resuming from checkpoints and saving relevant metadata (`classes`, `config`, etc.)

---

## Adding a New Method

To add a new method to MSD:
1. Create a model class that inherits from `AbstractModel`.
2. Create a trainer class that inherits from `AbstractTrainer`.
3. Optionally, implement a custom `LatentExplorer` if your model provides structure in the latent space (e.g., explicit static/dynamic separation).
4. Define a configuration file that includes the model, trainer, training parameters, and evaluation setup.
5. Reference your classes and configuration paths in the main YAML config file.

---

## Latent Explorer (Optional)

If your model provides architectural insights into its latent space (e.g., an explicit static/dynamic separation), you can define a custom `LatentExplorer` to leverage this knowledge during evaluation.

Refer to [docs/latent_exploration.md](latent_exploration.md) for implementation guidance and integration examples.

---

## Summary

The MSD benchmark separates modeling, training, and evaluation into pluggable modules. By adhering to the `AbstractModel` and `AbstractTrainer` interfaces, you can integrate new learning methods with minimal friction.

To see a working example, check the included methods directory. For experiment setup and configuration, refer to [docs/configuration.md](configuration.md).
