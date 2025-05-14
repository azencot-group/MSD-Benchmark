import json
from os import path as osp

import h5py
import numpy as np

from msd.utils.loading_utils import init_directories


class SpritesGenerator:
    """
    Converts the Sprites dataset into the DisentanglementDataset format.

    Combines static attributes and dynamic action labels into a single HDF5 file,
    including train/val/test splits, per-sample labels, and factor metadata.
    """

    def __init__(self, sprites_dir: str, output_path: str):
        """
        :param sprites_dir: Path to the directory containing npy files (should include 'npy/').
        :param output_path: Path to save the output .h5 file.
        """
        self.sprites_dir = sprites_dir
        self.output_path = output_path

    def initialize(self, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15):
        """
        Process and export the full dataset into .h5 format.

        :param seed: Random seed for shuffling.
        :param val_size: Proportion of data for validation.
        :param test_size: Proportion of data for test.
        """
        X_train, X_test, A_train, A_test, D_train, D_test = sprites_act(self.sprites_dir, return_labels=True)
        names = ['movement', 'body', 'bottom', 'top', 'hair']
        X = np.concatenate((X_train, X_test), axis=0)
        A = np.concatenate((A_train, A_test), axis=0)
        D = np.concatenate((D_train, D_test), axis=0)
        idx = np.arange(X.shape[0])
        rng = np.random.default_rng(seed=seed)
        idx = rng.permutation(idx)
        X, A, D = X[idx], A[idx], D[idx]
        data = (X * 256).astype(np.uint8).transpose(0, 1, 4, 2, 3)
        static = np.argmax(A[:, 0, :, :], axis=2)
        dynamic = np.expand_dims(np.argmax(D[:, 0, :], axis=1), 1)
        labels = np.concatenate((dynamic, static), axis=1)
        values = {
            'movement': {
            "front walk": 0,
            "left walk": 1,
            "right walk": 2,
            "front spellcard": 3,
            "left spellcard": 4,
            "right spellcard": 5,
            "front slash": 6,
            "left slash": 7,
            "right slash": 8
        }, 'body': {
            "light": 0,
            "neutral": 1,
            "dark-gray": 2,
            "gray": 3,
            "brown": 4,
            "black": 5
        }, 'bottom': {
            "white": 0,
            "yellow": 1,
            "red": 2,
            "gray": 3,
            "shade-green": 4,
            "green": 5
        }, 'top': {
            "red": 0,
            "blue": 1,
            "white": 2,
            "gray": 3,
            "brown": 4,
            "white-tie": 5
        }, 'hair': {
            "green": 0,
            "blue": 1,
            "orange": 2,
            "white": 3,
            "red": 4,
            "purple": 5
        }}
        classes = {s: {'index': i,
                       'type': 'dynamic' if i == 0 else 'static',
                       'n_classes': int(labels[:, i].max() + 1),
                       'values': values[s],
                       'ignore': False} for i, s in enumerate(names)}
        N = X_train.shape[0] + X_test.shape[0]
        n1, n2 = int(N * (1 - val_size - test_size)), int(N * (1 - test_size))
        init_directories(osp.dirname(self.output_path))
        with h5py.File(self.output_path, 'w') as h5_file:
            h5_file.create_dataset('data', data=data, dtype=np.uint8)
            h5_file.create_dataset('labels', data=labels)
            for set_, idxs in zip(['train', 'val', 'test'], [np.arange(n1), np.arange(n1, n2), np.arange(n2, N)]):
                h5_file.create_dataset(f'{set_}_indices', data=idxs)
            classes_json = json.dumps(classes)
            h5_file.attrs['classes'] = classes_json



def sprites_act(path: str, seed: int = 0, return_labels: bool = False):
    """
    Load raw Sprites dataset from .npy files.

    Combines multiple actions and directions into unified train/test splits,
    with optional return of attributes and action indicators.

    :param path: Base path to sprites data (should contain `npy/` subdir).
    :param seed: Random seed for shuffling.
    :param return_labels: If True, also return static and dynamic labels.
    :return: Tuple of arrays (X_train, X_test, A_train, A_test, D_train, D_test) if return_labels,
             otherwise just (X_train, X_test).
    """
    directions = ['front', 'left', 'right']
    actions = ['walk', 'spellcard', 'slash']
    path = osp.join(path, 'npy')
    X_train = []
    X_test = []
    A_train = []
    A_test = []
    D_train = []
    D_test = []
    for act in range(len(actions)):
        for i in range(len(directions)):
            label = 3 * act + i
            x = np.load(osp.join(path, '%s_%s_frames_train.npy' % (actions[act], directions[i])))
            X_train.append(x)
            y = np.load(osp.join(path, '%s_%s_frames_test.npy' % (actions[act], directions[i])))
            X_test.append(y)
            if return_labels:
                a = np.load(osp.join(path, '%s_%s_attributes_train.npy' % (actions[act], directions[i])))
                A_train.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1
                D_train.append(d)

                a = np.load(osp.join(path, '%s_%s_attributes_test.npy' % (actions[act], directions[i])))
                A_test.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1
                D_test.append(d)

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    np.random.seed(seed)
    ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[ind]
    if return_labels:
        A_train = np.concatenate(A_train, axis=0)
        D_train = np.concatenate(D_train, axis=0)
        A_train = A_train[ind]
        D_train = D_train[ind]
    ind = np.random.permutation(X_test.shape[0])
    X_test = X_test[ind]
    if return_labels:
        A_test = np.concatenate(A_test, axis=0)
        D_test = np.concatenate(D_test, axis=0)
        A_test = A_test[ind]
        D_test = D_test[ind]

    if return_labels:
        return X_train, X_test, A_train, A_test, D_train, D_test
    else:
        return X_train, X_test

if __name__ == '__main__':
    sprites_dir = '/path/to/sprites/'
    out_path = '/path/to/output/sprites.h5'

    sprites_dir = '/cs/cs_groups/azencot_group/MSD/datasets/sprites'
    out_path = '/cs/cs_groups/azencot_group/MSD/datasets/sprites/sprites/sprites_dataset.h5'

    generator = SpritesGenerator(sprites_dir, out_path).initialize(42, 0.15, 0.15)