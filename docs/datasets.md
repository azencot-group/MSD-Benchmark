# MSD Benchmark: Datasets Overview

The Multi-factor Sequential Disentanglement (MSD) benchmark supports a flexible and extensible dataset interface designed for disentangling static and dynamic factors in sequential data (video, audio, image sequences).

This document describes:
- The core dataset abstraction (`AbstractReader`)
- Supported dataset types
- Preprocessing hooks
- Dataset wrapper API
- Visualization utilities
- Adding new datasets

---

## Dataset Abstraction

All datasets in MSD are built on the `AbstractReader` base class, which provides a unified interface for reading sequences and their associated factor labels.

Each sample is represented as:
```python
x, static_factors, dynamic_factors = dataset[i]
```

Where:
- `x`: Input sequence (e.g., video frames or audio)
- `static_factors`: Dictionary of static factor labels
- `dynamic_factors`: Dictionary of dynamic factor labels

The `AbstractReader` provides:
- Metadata about each factor (type, class count, name mapping)
- Access to the label space via `classes`, `class_dims`, and `names_map`
- Filtering between static and dynamic factors

---

## Reader Implementations

### Hdf5Reader
For synthetic or preprocessed datasets stored in HDF5 format. Loads image or audio arrays and associated labels with minimal overhead.

### FileReader (Base class)
Abstract class for datasets where samples are loaded from file paths. Supports:
- Custom file reading via `_read(file_path)`
- Split filtering using a metadata CSV or DataFrame

#### ImageSequenceReader
Reads directories of image files as temporal sequences. Applies optional resizing, cropping, and normalization.

#### AudioReader
Loads raw waveform audio files and pads or truncates them to a fixed duration.

#### HuggingFaceReader
Loads datasets directly from Hugging Face Hub repositories. Supports secure access using a token file.

---

## Dataset Wrapper: DisentanglementDataset

The `DisentanglementDataset` class wraps any `AbstractReader` and provides:
- Optional supervised/unsupervised access
- Preprocessing hooks for image/audio transformation
- Human-readable label mapping (via `return_names=True`)

### Example Usage:
```python
from msd.data.datasets.disentanglement_dataset import DisentanglementDataset

reader = Hdf5Reader("/path/to/data.h5", split="train")
dataset = DisentanglementDataset(reader, supervised=True, return_names=True)

x, ys, yd = dataset[0]  # image, static labels, dynamic labels
```

---

## Preprocessing Hooks

Preprocessing hooks allow easy, pluggable transformations at data load time. Each hook implements an `apply(data)` method and can be composed in sequence.

Supported hooks:
- `Transpose(axes)`: Reorder dimensions
- `Normalize(min_val, max_val)`: Rescale input range
- `ToNumpy()`: Convert tensor-like input to NumPy arrays

Example:
```python
dataset = DisentanglementDataset(
    reader,
    preprocess_hooks=[Transpose((1, 2, 0)), Normalize(), ToNumpy()],
    supervised=True
)
```

---

## Visualization Utilities

To aid debugging and result interpretation, MSD provides built-in plotting tools:

- `plot_image(image)` – Show single (C, H, W) image
- `plot_sequence(sequence)` – Display sequence of frames side by side
- `plot_sequences(list_of_sequences)` – Multiple sequences in rows
- `compare_sequences(set1, titles1, set2, titles2)` – Side-by-side comparison
- `create_video(sequence, out_path)` – Save a sequence as a .mp4 video

---

## Adding a Dataset

To add a new dataset to MSD, you need to provide:

1. **A reader class** that conforms to the `AbstractReader` interface. You can either implement a custom reader or adapt your files to match an existing reader format (e.g., HDF5, directory-based image/audio files).

2. **A `classes.json` file**, which defines the factor metadata. This dictionary should include, for each factor:
   - `index`: Column index in the label array
   - `type`: Either `'static'` or `'dynamic'`
   - `n_classes`: Number of unique labels
   - `ignore`: Whether to exclude this factor from evaluation
   - `values`: A dictionary mapping human-readable names to numeric indices

### Example - HDF5 File Format
If using the `Hdf5Reader`, your `.h5` file should contain:
- `'classes'` (attribute): JSON-encoded string matching the `classes.json` file
- `'data'` (dataset): NumPy array of shape `(N, D)` where `D` is the sample shape (e.g., `T x C x H x W` for video)
- `'labels'` (dataset): NumPy array of shape `(N, L)`, where each row contains label indices per factor
- `'<split>_indices'` (dataset): NumPy array for each split name (`train_indices`, `val_indices`, `test_indices`) containing sample indices

For a complete walkthrough of how to define, animate, export and load a synthetic dataset using MSD tools, refer to the tutorial notebook:

📄 [`examples/synthetic_dsprites_tutorial.ipynb`](../examples/synthetic_dsprites_tutorial.ipynb)

---

## Summary

The MSD dataset layer abstracts over input formats and modality-specific quirks, while giving fine control over factor labeling, sequence handling, and preprocessing. New datasets can be integrated by subclassing `AbstractReader`, or extending any existing reader class to fit your data structure.