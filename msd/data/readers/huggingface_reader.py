import json
from pathlib import Path
from os import path as osp
from typing import Union, List, Dict, Tuple, Optional

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from msd.data.readers.abstract_reader import AbstractReader


class HuggingFaceReader(AbstractReader):
    def __init__(self, repo_id: str, split: str, token: Optional[str] = None):
        """
        Initialize the HuggingFaceReader with the repository ID, split, and token.

        :param repo_id: Repository ID on Hugging Face.
        :param split: Split to load ('train', 'val', or 'test').
        :param token: Token for authentication. If None, will attempt retrieve token from .hf/token.txt.
        """
        self.repo_id = repo_id
        if token is None:
            token = Path.home().joinpath('.hf', 'token.txt')
            if osp.exists(token):
                token = token.read_text().strip()
        classes_path = hf_hub_download(repo_id=self.repo_id, filename="classes.json", repo_type="dataset", token=token)
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        super().__init__(classes, split)

        self._dataset = load_dataset(self.repo_id, token=token)
        self.dataset = self._dataset[self.split]

    def __len__(self) -> int:
        """
        :return: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: Union[int, slice, List]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve sample(s) from the dataset.
        :param index: Index of the sample to retrieve.
        :return: Tuple containing the data, static factors, and dynamic factors.
        """
        if isinstance(index, int):
            index = [index]
        data = self.dataset[index]
        static_factors = {k: np.array(data[k]).squeeze() for k in self.static_factors}
        dynamic_factors = {k: np.array(data[k]).squeeze() for k in self.dynamic_factors}
        return data['x'], static_factors, dynamic_factors

    def __repr__(self) -> str:
        """
        :return: String representation of the HuggingFace reader.
        """
        return f'{self.__class__.__name__}(split={self.split}, classes={set(self.classes.keys())}, repo_id={self.repo_id})'