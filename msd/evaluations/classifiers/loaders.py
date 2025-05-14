from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from os import path as osp

import torch
from huggingface_hub import hf_hub_download

from msd.configurations.msd_component import MSDComponent


class ClassifierLoader(ABC, MSDComponent):
    """
    Base class for loading classifiers.
    """
    def __init__(self, checkpoint_path: str):
        """
        Initialize the ClassifierLoader with the path to the checkpoint file.
        :param checkpoint_path: Path to the checkpoint file.
        """
        self.checkpoint_path = checkpoint_path

    def load_classifier(self):
        return torch.load(self.checkpoint_path, weights_only=False)

class HuggingfaceLoader(ClassifierLoader):
    """
    Loads a classifier from Huggingface.
    """

    def __init__(self, repo_id: str, repo_path: str, token: Optional[str] = None):
        """
        Initialize the HuggingfaceLoader with the repository ID and path to the checkpoint file.
        :param repo_id: Repository ID on Hugging Face.
        :param repo_path: Path to the checkpoint file in the repository.
        :param token: Token for authentication. If None, will attempt to retrieve token from .hf/token.txt.
        """
        self.repo_id = repo_id
        self.repo_path = repo_path
        if token is None:
            token = Path.home().joinpath('.hf', 'token.txt')
            if osp.exists(token):
                token = token.read_text().strip()
        checkpoint_path = hf_hub_download(repo_id=self.repo_id, repo_type='model', filename=self.repo_path, token=token)
        super().__init__(checkpoint_path)
