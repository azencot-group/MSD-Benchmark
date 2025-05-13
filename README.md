# MSD Benchmark: Multi-Factor Sequential Disentanglement

MSD is a benchmark for evaluating disentangled representation learning on **sequential data** (e.g., videos, audio, time-series). It supports datasets with both static and dynamic factors, and includes tools for automatic annotation, model evaluation, and visualization.

![Overview](figures/benchmark_overview.png)

---

## 🛠️ Installation

We recommend using a conda environment:

```bash
conda create -n msd python=3.9
conda activate msd
pip install -r requirements.txt
```

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

All training and evaluation is handled by the `run.py` script.

### Train a model:

```bash
python run.py --run_config configurations/methods/ssm_skd/ssm_skd_sprites.yaml --train
```

### Evaluate a model:

```bash
python run.py --run_config configurations/methods/ssm_skd/ssm_skd_sprites.yaml --eval
```

You must also specify a `meta.yaml` file with global variables like `msd_root`. This is optional if your `meta.yaml` is located at `configurations/meta.yaml`.

---

## 🧠 Automatic Annotation

Use `auto_annotate.py` to automatically discover and label factor spaces in new datasets using a vision-language model.

```bash
python auto_annotate.py \
  --ds_path /path/to/dataset.h5 \
  --subset train \
  --n_exploration 500 \
  --n_annotation 500 \
  --out_dir /path/to/output \
  --ds_name my_dataset
```

For details, see [docs/vlm_module.md](docs/vlm_module.md)

---

## 📎 Citation

TBD.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact the authors.