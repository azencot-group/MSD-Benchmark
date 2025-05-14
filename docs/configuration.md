
# MSD Benchmark: Configuration System

The MSD (Multi-factor Sequential Disentanglement) benchmark is driven by a flexible, hierarchical configuration system. All training and evaluation executions are defined through structured YAML files, enabling consistent and reproducible experimentation.

This document explains:
- The role of the configuration system
- Structure and dependencies between configuration files
- Key configuration components and examples

---

## Overview

The MSD configuration system allows users to:
- Select datasets, models, trainers, and evaluators
- Define training hyperparameters and evaluation behavior
- Organize configuration files by dataset, method, evaluation mode, and global scope

Execution is governed by a root configuration file that ties together all components. Most parameters are automatically injected through interpolation (e.g., `${name}`, `${out_dir}`) for maintainability.

---

## ConfigInitializer

At the core of the configuration system lies the `ConfigInitializer` utility. It is responsible for parsing the full configuration file and instantiating all required MSD components including:
- Datasets and data loaders
- Models
- Trainers
- Evaluators
- Loggers

Any `Trainer` or `Evaluator` class receives a `ConfigInitializer` instance as its constructor argument. This allows these components to access all relevant sections of the configuration file and initialize their dependencies with minimal boilerplate code.

---

## Configuration Hierarchy

### 1. `meta.yaml` (Global Settings)
Defines benchmark-wide settings such as:
- `msd_root`: Root directory for outputs
- Random seed or logging formats (optional)

This file is referenced by other configurations using `${msd_root}`.

---

### 2. Dataset Configuration (`configurations/datasets/*.yaml`)
Each dataset (e.g., `sprites`, `shapes3d`) has its own configuration specifying:
- Dataset name and variant (e.g., `dsprites`, `dsprites_static`)
- Data type (video, audio, or timeseries)
- Reader class (e.g., `Hdf5Reader`, `HuggingFaceReader`)
- Preprocessing hooks
- Train/val/test data loaders

For a full overview of dataset readers, formats, and supported preprocessing hooks, refer to [docs/datasets.md](datasets.md).

**Example: `sprites.yaml`**
```yaml
dataset_name: sprites
dataset_variant: sprites
data_type: 'Video'
classifier_loader_cfg:
  name: HuggingfaceLoader
  parameters:
    repo_id: "TalBarami/msd_judge_classifiers"
    repo_path: "${dataset_variant}_classifier_best.pth"
dataset:
  dataset:
    name: DisentanglementDataset
    parameters: {}
  reader:
    name: HuggingFaceReader
    parameters:
      repo_id: "TalBarami/msd_${dataset_variant}"
  preprocess_hooks:
    - name: ToNumpy
      parameters: {}
    - name: Transpose
      parameters:
        axes: [0, 3, 1, 2]
    - name: Normalize
      parameters:
        data_min: 0
        data_max: 255
  splits:
    train:
      preprocess_hooks: ${dataset.preprocess_hooks}
      loader:
        name: DataLoader
        parameters:
          batch_size: ${batch_size}
          shuffle: true
          pin_memory: true
          drop_last: true
    val:
      preprocess_hooks: ${dataset.preprocess_hooks}
      loader:
        name: DataLoader
        parameters:
          batch_size: ${test_batch_size}
          shuffle: false
          pin_memory: true
          drop_last: true
    test:
      preprocess_hooks: ${dataset.preprocess_hooks}
      loader:
        name: DataLoader
        parameters:
          batch_size: ${test_batch_size}
          shuffle: false
          pin_memory: true
          drop_last: true
```

---

### 3. Evaluation Configuration (`configurations/evaluations/*.yaml`)
Used both during training and testing. These files define:
- Judge and predictor used to probe latent space
- Evaluation metrics and visualization tools
- Logging behavior

For details on the evaluation manager and evaluators, see [docs/evaluation_metrics.md](evaluation_metrics.md).

**Example: `video_training_evaluation.yaml`**
```yaml
seed: 42
evaluation:
  evaluation_manager:
    name: EvaluationManager
    parameters:
      main: 0
      repeat: 1
      dataset_type: val
      judge_cfg:
        name: ClassifierJudge
        parameters:
          classifier_cfg:
            name: ${data_type}Classifier
            parameters: {}
          classifier_loader_cfg: ${classifier_loader_cfg}
      predictor_cfg:
        name: GradientBoostingClassifier
        parameters:
          max_depth: 3
      latent_explorer_cfg: ${latent_explorer_cfg}
  evaluators:
    - name: MultiFactorSample
      parameters: {}
    - name: TwoFactorSample
      parameters: {}
    - name: VideoReconstruction
      parameters:
        n_samples: 4
    - name: LatentSwapVisualizer
      parameters: {}
loggers:
  - name: FileLogger
    parameters:
      name: ${name}
      show: false
      log_path: ${out_dir}
```

---

### 4. Method-Specific Configuration
Each method and dataset pair gets its own configuration. These files specify:
- Model architecture and latent explorer config
- Trainer class and loss weights
- Paths to datasets and evaluation configs

For model integration details and implementation guidance, see [docs/methods.md](methods.md).

See also [docs/latent_exploration.md](latent_exploration.md) for details on latent explorer configuration.

**Example: `ssm_skd_sprites.yaml`**
```yaml
model_name: ssm_skd
name: ${model_name}_${dataset_variant}
out_dir: ${msd_root}/models/${name}
checkpoint_dir: ${out_dir}
device: cuda
load_model: best

dataset: configurations/datasets/sprites.yaml
training_evaluation: configurations/evaluations/video_training_evaluation.yaml
testing_evaluation: configurations/evaluations/video_testing_evaluation.yaml

batch_size: 32
test_batch_size: 1024
latent_explorer_cfg:
  name: PredictorLatentExplorer
  parameters:
    batch_exploration: true

model:
  name: SsmSkd
  parameters:
    in_dim: 3
    k_dim: 15
    hidden_dim: 170
    data_type: video

trainer:
  name: SsmSkdTrainer
  save_model: true
  parameters:
    w_rec: 16.0
    w_pred: 1.0
    w_eigs: 1.0
    dynamic_thresh: 0.75
    gradient_clip_val: 5
  resume: false
  scheduler:
    name: StepLR
    parameters:
      step_size: 250
      gamma: 0.5
  optimizer:
    name: Adam
    parameters:
      lr: 0.001
  epochs:
    start: 1
    end: 250
  verbose: 10
```

---

## Summary

The configuration system in MSD separates responsibilities cleanly:
- Global (`meta.yaml`) for root paths and constants
- Dataset-level for loading and preprocessing
- Evaluation-level for metrics and probes
- Method-specific for models, trainers, and hyperparameters

This design enables reusable components and fast experimentation. See template files in the `configurations/` directory to get started.
