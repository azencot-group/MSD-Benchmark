import json
from typing import Union, List, Dict, Tuple

import h5py
import numpy as np

from msd.data.readers.abstract_reader import AbstractReader


class Hdf5Reader(AbstractReader):
    def __init__(self, h5_path: str, split: str):
        """
        Initialize the Hdf5Reader with the path to the HDF5 file and the desired split.

        :param h5_path: Path to the HDF5 file.
        :param split: Split to load ('train', 'val', or 'test').
        """
        self.h5_path = h5_path

        self.h5_file = h5py.File(h5_path, 'r')
        classes_json = self.h5_file.attrs['classes']
        classes = json.loads(classes_json)
        super().__init__(classes, split)

        self.data = self.h5_file['data'] # NxTx(D) , (D) = CxHxW for video
        self.labels = self.h5_file['labels'] # NxL
        self.indices = self.h5_file[f'{split}_indices'][:] # N(split)

    def __len__(self) -> int:
        """
        :return: Number of samples in the dataset split.
        """
        return len(self.indices)

    def get_factors(self, y: np.ndarray, ftype: str) -> Dict[str, np.ndarray]:
        """
        Retrieve factors of a specific type (static or dynamic) from the labels.

        :param y: Labels array.
        :param ftype: Type of factors to retrieve ('static' or 'dynamic').
        :return: Dictionary of factors.
        """
        return {k: y[:, v['index']].squeeze() for k, v in self.classes.items() if v['type'] == ftype}

    def __getitem__(self, index: Union[int, slice, List]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve sample(s) from the dataset.
        :param index: Index of the sample to retrieve.
        :return: Tuple containing the data, static factors, and dynamic factors.
        """
        if isinstance(index, int):
            index = [index]
        data_index = self.indices[index]
        # H5 files require increasing order of indices
        pos, i = zip(*sorted(enumerate(data_index), key=lambda x: x[1]))
        pos, i = np.array(pos), np.array(i)
        # Return to the original order
        x = self.data[i][pos].squeeze()
        y = self.labels[i][pos]
        static_factors = self.get_factors(y, 'static')
        dynamic_factors = self.get_factors(y, 'dynamic')

        return x, static_factors, dynamic_factors

    def close(self):
        """
        Close the HDF5 file.
        """
        self.h5_file.close()

    def __repr__(self) -> str:
        """
        :return: String representation of the HDF5 reader.
        """
        return f'{self.__class__.__name__}(split={self.split}, classes={set(self.classes.keys())}, h5_path={self.h5_path})'


