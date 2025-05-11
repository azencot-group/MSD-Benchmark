import os
from os import path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

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

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    dmw_root = r'/cs/cs_groups/azencot_group/MSD/datasets/dmelodies_wav'
    files_info = r'/cs/cs_groups/azencot_group/MSD/datasets/dmelodies_wav/files_info.csv'
    classes_file = r'/cs/cs_groups/azencot_group/MSD/datasets/dmelodies_wav/classes.json'
    dmw = dMelodiesWAVProcess(dmw_root)
    if osp.exists(files_info) and osp.exists(classes_file):
        df = pd.read_csv(files_info)
        classes = read_json(classes_file)
    else:
        df = dmw.create_files_info()
        classes = dmw.create_classes(df)
        df.to_csv(files_info, index=False)
        write_json(classes_file, classes)

    reader = AudioReader(
        files_root=dmw_root,
        files_info=df,
        split='train',
        classes=classes,
        sample_rate=dmw.sample_rate
    )
    for i, x in enumerate(reader):
        continue