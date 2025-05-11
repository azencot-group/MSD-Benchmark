import json
import pickle
from pathlib import Path
from typing import Union, Any

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig


def init_directories(*dirs: Union[str, Path]) -> None:
    """
    Create directories if they do not exist.

    :param dirs: Any number of directory paths.
    """
    for dir_ in dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)


def save_config(config: DictConfig, out: str) -> None:
    """
    Save an OmegaConf config to disk as a YAML file.

    :param config: The OmegaConf configuration to save.
    :param out: Path to output file.
    """
    with open(out.replace('\\', '/'), 'w') as fp:
        OmegaConf.save(config=config, f=fp.name)


def load_config(file: str, meta_file: str, train_mode: bool, resolve: bool = True) -> DictConfig:
    """
    Load and resolve a hierarchical config, merging dataset and evaluation configs.

    :param file: Path to the main config file.
    :param meta_config: Path to the meta config file.
    :param train_mode: If True, load training_evaluation config; otherwise load testing_evaluation.
    :param resolve: Whether to resolve interpolations in OmegaConf.
    :return: Merged and resolved DictConfig.
    """
    with open(meta_file.replace('\\', '/'), 'r') as fp:
        meta_cfg = OmegaConf.load(fp.name)
    with open(file.replace('\\', '/'), 'r') as fp:
        cfg = OmegaConf.load(fp.name)
    cfg = OmegaConf.merge(meta_cfg, cfg)
    if type(cfg.dataset) is str:
        data_cfg = OmegaConf.load(cfg.dataset)
        cfg = OmegaConf.merge(cfg, data_cfg)
    if 'evaluation' not in cfg:
        eval_cfg = OmegaConf.load(cfg.training_evaluation if train_mode else cfg.testing_evaluation)
    elif type(cfg.evaluation) is str:
        eval_cfg = OmegaConf.load(cfg.evaluation)
    else:
        eval_cfg = {}
    cfg = OmegaConf.merge(eval_cfg, cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=resolve))
    return cfg

def read_pkl(file: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.

    :param file: Path to pickle file.
    :return: Loaded Python object.
    """
    with open(file, 'rb') as f:
        return pickle.load(f)

def write_pkl(file: Union[str, Path], data: Any) -> None:
    """
    Write an object to a pickle file.

    :param file: Path to output file.
    :param data: Python object to serialize.
    """
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_json(file: Union[str, Path]) -> Any:
    """
    Read a JSON file.

    :param file: Path to the JSON file.
    :return: Parsed Python object (usually dict or list).
    """
    with open(file, 'r') as f:
        return json.load(f)


def write_json(file: Union[str, Path], data: Any) -> None:
    """
    Write a Python object to a JSON file.

    :param file: Path to save to.
    :param data: JSON-serializable object.
    """
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy and torch.

    :param seed: Integer seed value.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

