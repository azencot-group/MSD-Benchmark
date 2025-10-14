import os
from os import path as osp

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from msd.data.readers.file_readers import AudioReader
from msd.utils.loading_utils import read_json, write_json


class dMelodiesWAVProcess:
    def __init__(self, dmelodies_wav_root: str):
        self.dmw_dir = dmelodies_wav_root
        self.static, self.dynamic = ['instrument'], ['tonic', 'octave', 'scale', 'rhythm_bar1', 'arp_chord1', 'arp_chord2']
        self.label_cols = self.static + self.dynamic
        self.sample_rate = 16000
        self.duration_seconds = 3

    def create_files_info(self, seed=42, test_size=0.15, val_size=0.15):
        df = pd.DataFrame(columns=['filename', 'file_path'] + self.label_cols)
        instruments = [d for d in os.listdir(self.dmw_dir) if osp.isdir(osp.join(self.dmw_dir, d))]
        for instrument in tqdm(instruments, desc="Processing Instruments"):
            for f in tqdm(os.listdir(osp.join(self.dmw_dir, instrument)), desc=f"Processing Files in {instrument}", leave=False):
                label = f.split('.')[0].split('_')[1:]
                df.loc[df.shape[0]] = [f, osp.join(instrument, f)] + [instrument] + label
        df[['octave', 'rhythm_bar1']] = df[['octave', 'rhythm_bar1']].astype(int)
        df['idx'] = df['filename'].apply(lambda f: f.split('_')[0]).astype(int)
        df.sort_values(by=['instrument', 'idx']).reset_index(drop=True)

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(df.index)
        df['split'] = 'train'
        n1, n2 = int(df.shape[0] * (1 - val_size - test_size)), int(df.shape[0] * (1 - test_size))
        df.loc[shuffled_indices[n1:], 'split'] = 'val'
        df.loc[shuffled_indices[n2:], 'split'] = 'test'

        return df

    def create_classes(self, df):
        values_map = {}
        for c in self.label_cols:
            le = LabelEncoder()
            df[f'{c}_value'] = df[c]
            df[c] = le.fit_transform(df[c]).astype(int)
            values_map[c] = dict(zip([int(_x) for _x in le.transform(le.classes_)], le.classes_))
        classes = {s: {'index': i,
                       'type': 'static' if s in self.static else 'dynamic',
                       'n_classes': len(values_map[s]),
                       'ignore': len(values_map[s]) == 1,
                       'values': {int(v) if type(v) == np.int64 else v: int(k) for k, v in values_map[s].items()}} for i, s in enumerate(self.label_cols)}
        return classes

def initialize_dmelodies_wav(files_dir, df_path, classes_path, seed=42, test_size=0.15, val_size=0.15):
    """
    Initialize the dMelodiesWAV dataset by creating the files info dataframe and the classes.json file if they do not exist.
    :param files_dir: Path to the directory containing the dMelodiesWAV files.
    :param df_path: Path to the CSV file containing the files info.
    :param classes_path: Path to the JSON file containing the classes info.
    :param seed: Random seed for shuffling the data.
    :param test_size: Proportion of the data to be used for testing.
    :param val_size: Proportion of the data to be used for validation.
    :return:
    """
    dmw = dMelodiesWAVProcess(files_dir)
    if osp.exists(df_path) and osp.exists(classes_path):
        df = pd.read_csv(df_path)
        classes = read_json(classes_path)
    else:
        df = dmw.create_files_info(seed=seed, test_size=test_size, val_size=val_size)
        classes = dmw.create_classes(df)
        df.to_csv(df_path, index=False)
        write_json(classes_path, classes)

    # Initialize the AudioReader with the files_root, files_info, split, classes, and sample_rate to access the audio files
    reader = AudioReader(
        files_root=files_dir,
        files_info=df,
        split='train',
        classes=classes,
        sample_rate=dmw.sample_rate
    )
    return reader

if __name__ == '__main__':
    # Example usage
    dmelodies_wav_root = '/path/to/dmelodies_wav'
    files_info = '/path/to/create/files_info.csv'
    classes_file = '/path/to/create/classes.json'

    reader = initialize_dmelodies_wav(dmelodies_wav_root, files_info, classes_file)

    # Example of reading audio data
    for i, x in enumerate(reader):
        print(f"Audio sample {i}: {x}")
        if i >= 5:  # Limit to first 5 samples for demonstration
            break
