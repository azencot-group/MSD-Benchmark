import os
from os import path as osp
from abc import abstractmethod
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from scipy.io import wavfile

from msd.data.readers.abstract_reader import AbstractReader

class FileReader(AbstractReader):
    def __init__(self, files_root: str, files_info: Union[str, pd.DataFrame], split: str, classes: Dict[str, Dict[str, Any]]):
        """
        Initialize the FileReader with the path to the files and the desired split.

        :param files_root: Path to the root directory containing the files.
        :param files_info: Path to a CSV file or a DataFrame containing file information. Specifically, it should contain the following keys: ['file_path', 'split'].
        :param split: Split to load ('train', 'val', or 'test').
        :param classes: Contains class information.
        """
        super().__init__(classes, split)
        self.files_root = files_root
        if isinstance(files_info, str):
            files_info = pd.read_csv(files_info, index_col=0)
        self.files_info = files_info

        self.df = self.files_info[self.files_info['split'] == self.split].copy().reset_index(drop=True)
        self.df['file_path'] = self.df['file_path'].apply(lambda p: osp.join(self.files_root, p))

    def __len__(self) -> int:
        """
        :return: Number of samples in the dataset split.
        """
        return len(self.df)

    def __getitem__(self, index: Union[int, slice, List]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Retrieve sample(s) from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing the data, static factors, and dynamic factors.
        """
        if isinstance(index, int):
            index = [index]
        if any(i >= self.__len__() for i in index):
            raise IndexError(f"Index {index} out of range for dataset of length {self.__len__()}.")
        df = self.df.loc[index]
        static = {col: df[col].to_numpy().squeeze() for col in self.static_factors}
        dynamic = {col: df[col].to_numpy().squeeze() for col in self.dynamic_factors}
        data = np.array([self._read(x) for x in df['file_path'].values]).squeeze()

        return data, static, dynamic

    @abstractmethod
    def _read(self, file_path):
        """
        Read the file and return its content.

        :param file_path: Path to the file.
        :return: Content of the file.
        """
        pass

class ImageSequenceReader(FileReader):
    def __init__(self, files_root: str, files_info: Union[str, pd.DataFrame], split: str, classes: Dict[str, Dict[str, Any]],
                 image_size=64, sequence_length=10):
        """
        Initialize the ImageSequenceReader with the path to the files and the desired split.
        :param files_root: Path to the root directory containing the files.
        :param files_info: Path to a CSV file or a DataFrame containing file information.
        :param split: Split to load ('train', 'val', or 'test').
        :param classes: Contains class information.
        :param image_size: Transform image size.
        :param sequence_length: Transform sequence length.
        """
        super().__init__(files_root, files_info, split, classes)
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            Image.open,
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])

    def _read(self, files_dir):
        """
        Read the image sequence from the directory and return it as a numpy array.
        :param files_dir: Path to the directory containing the image sequence.
        :return: Numpy array of the image sequence.
        """
        files = sorted([f for f in os.listdir(files_dir)])
        sequence_length = len(files)
        s = np.random.randint(0, sequence_length - self.sequence_length)
        t = s + self.sequence_length

        images = [self.transform(os.path.join(files_dir, f)) for f in files[s:t]]
        images = np.stack(images)
        return images

class AudioReader(FileReader):
    def __init__(self, files_root: str, files_info: Union[str, pd.DataFrame], split: str, classes: Dict[str, Dict[str, Any]], sample_rate=16000, seconds=3):
        """
        Initialize the AudioSequenceReader with the path to the files and the desired split.
        :param files_root: Path to the root directory containing the files.
        :param files_info: Path to a CSV file or a DataFrame containing file information.
        :param split: Split to load ('train', 'val', or 'test').
        :param classes: Contains class information.
        :param sample_rate: Sample rate for audio files.
        """
        super().__init__(files_root, files_info, split, classes)
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.T = self.seconds * self.sample_rate

    def _read(self, file_path):
        """
        Read the audio file and return its content.
        :param file_path: Path to the audio file.
        :return: Content of the audio file.
        """
        _, d = wavfile.read(file_path)
        d = np.pad(d, (0, self.T - d.shape[0]), 'constant')
        return d
