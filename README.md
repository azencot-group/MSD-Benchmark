# MSD Benchmark: Multi-Factor Sequential Disentanglement

MSD is a benchmark for evaluating disentangled representation learning on **sequential data** (e.g., video, audio, time series). It supports datasets with both static and dynamic factors, and includes tools for automatic annotation, model evaluation, and visualization.

![Overview](figures/benchmark_overview.png)

---

## 🛠️ Installation

We recommend using a conda environment:

```bash
conda create -n msd python=3.9.21
conda activate msd
pip install -r requirements.txt
```

After installation, edit the `configurations/meta.yaml` file and set the `msd_root` field to the absolute path where all outputs (e.g., logs, models, results) should be saved.

---

## 📂 Components

MSD is modular and consists of the following major components:

- **Datasets**: Loaders for synthetic and real-world data. See [docs/datasets.md](docs/datasets.md)
- **Methods**: Plug-and-play training pipelines for disentanglement models. See [docs/methods.md](docs/methods.md)
- **Latent Exploration**: Analyzes latent codes to discover factor-aligned dimensions. See [docs/latent_exploration.md](docs/latent_exploration.md)
- **Evaluation**: Metrics and visualizations for judging model performance. See [docs/evaluation_metrics.md](docs/evaluation_metrics.md)
- **VLM Tools**: Use vision-language models (e.g., GPT-4) for automatic annotation and factor classification. See [docs/vlm_module.md](docs/vlm_module.md)
- **Configuration**: YAML-driven setup for all runs. See [docs/configuration.md](docs/configuration.md)

---

## 🚀 Running Experiments

All training and evaluation is handled by the `msd/run.py` script.

### Train a model:

```bash
PYTHONPATH="." python msd/run.py --run_config configurations/methods/ssm_skd/ssm_skd_sprites.yaml --train
```

### Evaluate a model:

```bash
PYTHONPATH="." python msd/run.py --run_config configurations/methods/ssm_skd/ssm_skd_sprites.yaml --eval
```
The model will be automatically loaded from the `checkpoint_dir` path specified in the configuration file.

### Model Zoo:

See Hugging Face: [TalBarami/msd_models](https://huggingface.co/TalBarami/msd_models/tree/main)

---

## 🧠 Automatic Annotation

Use `msd/auto_annotate.py` to automatically discover and label factor spaces in new datasets using a vision-language model.

```bash
PYTHONPATH="." python msd/auto_annotate.py \
  --ds_path /path/to/dataset.h5 \
  --subset train \
  --n_exploration 500 \
  --n_annotation 500 \
  --out_dir /path/to/output \
  --ds_name my_dataset
```

For details, see [docs/vlm_module.md](docs/vlm_module.md)

---

## ⚙️ Reference Experimental Environment

| Component               | Authors'                           |
|-------------------------|------------------------------------|
| CPU                     | AMD EPYC 7002 Series Processor     |
| CPU architecture        | x86_64                             |
| Linux kernel version    | 5.14.0                             |
| glibc version           | 2.34                               |
| GPU                     | NVIDIA GeForce RTX 4090 (Gigabyte) |
| NVIDIA driver version   | 565.57.01                          |
| NVIDIA VBIOS version    | 95.02.3C.C0.93                     |
| CUDA version            | 12.6                               |
| cuDNN version           | 9.5.1                              |
| Python version          | 3.9.21                             |
| pip version             | 25.1                               |
| NumPy version           | 1.26.4                             |
| PyTorch version         | 2.7.0                              |
| Python package versions | `requirements.txt`                 |

---

## 📎 Citation

T. Barami, N. Berman, I. Naiman, A. H. Hason, R. Ezra, O. Azencot, "**Disentanglement Beyond Static vs. Dynamic: A Benchmark and Evaluation Framework for Multi-Factor Sequential Representations**" in NeurIPS 2025, Forthcoming.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact the authors.
