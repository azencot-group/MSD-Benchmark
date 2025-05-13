import json
from typing import Union, Dict
from os import path as osp

import h5py
import numpy as np
from tqdm import tqdm

from msd.data.datasets.synthetic_video_generation.factor_space import FactorSpace
from msd.data.datasets.synthetic_video_generation.state_mapper import StateMapper
from msd.utils.loading_utils import init_directories


class VideoGenerator:
    """
    A class to generate videos based on static and dynamic factors.
    This class utilizes the StateMapper class to map states to video frames.
    """
    def __init__(self, state_mapper: StateMapper, static_factors: FactorSpace, dynamic_factors: FactorSpace, T: int):
        """
        Initialize the VideoGenerator with a state mapper, static factors, dynamic factors, and the number of frames T.
        :param state_mapper: A StateMapper object that maps states to video frames.
        :param static_factors: A list of StaticFactor objects.
        :param dynamic_factors: A list of DynamicFactor objects.
        :param T: The number of frames in the video.
        """
        self.state_mapper = state_mapper
        self.static_factors = static_factors
        self.dynamic_factors = dynamic_factors
        self.factors = static_factors | dynamic_factors
        self.T = T
        self.state_space = self.factors.combinations()
        self.i = 0

    def __iter__(self):
        """
        Initialize the iterator.
        :return: self
        """
        self.i = 0
        return self

    def __next__(self):
        """
        Get the next video frame.
        :return: The next video frame.
        """
        if self.i >= len(self.state_space):
            raise StopIteration
        factors = self.state_space[self.i]
        self.i += 1
        return self.generate(factors)

    def __getitem__(self, factors: Union[int, Dict[str, int]]) -> np.ndarray:
        """
        Retrieve a video corresponding to given factor labels or index.
        :param factors: A dictionary with keys as factor names and values as their corresponding labels or an index.
        :return: Corresponding RGB video of shape (T, C, H, W), dtype=uint8.
        """
        return self.generate(factors)

    def __len__(self):
        """
        Get the number of videos in the dataset.
        :return: The number of videos.
        """
        return len(self.state_space)

    def generate(self, factors: Union[int, Dict[str, int]]) -> np.ndarray:
        """
        Generate a video corresponding to given factor labels.
        :param factors: A dictionary with keys as factor names and values as their corresponding labels.
        :return: Corresponding RGB video of shape (T, C, H, W), dtype=uint8.
        """
        if isinstance(factors, int):
            factors = self.state_space[factors]
        static = {k: self.static_factors[k].values[v] for k, v in factors.items() if k in self.static_factors}
        dynamic = {self.dynamic_factors[k].static_factor.name: self.dynamic_factors[k].sequences[v][:self.T] for k, v in factors.items() if k in self.dynamic_factors}

        seq = []
        for t in range(self.T):
            state = static | {k: v[t] for k, v in dynamic.items()}
            seq.append(self.state_mapper[state])
        return np.array(seq).astype(np.uint8)

    def create_dataset(self, out_path: str, seed: int = 42, val_size: int = 0.15, test_size: int = 0.15):
        """
        Create a dataset of videos with the corresponding state-space.
        :param out_path: Path to save the dataset.
        :param seed: Random seed for shuffling the dataset.
        :param val_size: Fraction of validation samples.
        :param test_size: Fraction of test samples.
        """
        N = len(self.state_space)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N)
        splits = {
            'train': indices[:int(N * (1 - val_size - test_size))],
            'val': indices[int(N * (1 - val_size - test_size)):int(N * (1 - test_size))],
            'test': indices[int(N * (1 - test_size)):]
        }

        classes = {s.name:
                       {'index': i,
                       'type': s.factor_type,
                       'n_classes': len(s),
                       'ignore': False,
                       'values': {k: int(v) for k, v in s.values_map.items()}}
                   for i, s in enumerate(self.factors)}


        init_directories(osp.dirname(out_path))
        with h5py.File(out_path, 'w') as h5_file:
            h5_file.create_dataset('data', shape=(N, *self[0].shape), dtype='uint8')
            h5_file.create_dataset('labels', shape=(N, len(self.factors)), dtype='int32')

            for i, factors in tqdm(enumerate(self.state_space)):
                label = [factors[k] for k in self.factors.names]
                sample = self.generate(factors)
                h5_file['data'][i] = sample
                h5_file['labels'][i] = label
            for split, idxs in splits.items():
                h5_file.create_dataset(f'{split}_indices', data=idxs)
            classes_json = json.dumps(classes)
            h5_file.attrs['classes'] = classes_json
        print(f'Dataset saved to {out_path}')
