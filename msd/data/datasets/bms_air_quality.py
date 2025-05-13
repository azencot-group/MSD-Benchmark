import json
import os
from os import path as osp

import h5py
import numpy as np
import pandas as pd

from msd.utils.loading_utils import init_directories

class BMSAirQualityPreprocess:
    """
    Converts the Beijing Multi-Site Air-Quality (BMS-AQ) dataset into the DisentanglementDataset format.

    Loads, normalizes, and processes per-day weather records, grouped by station and date,
    then exports a structured HDF5 file with labels and metadata.
    """

    def __init__(self, weather_dir: str, output_path: str):
        """
        :param weather_dir: Root folder containing PRSA csv files.
        :param output_path: Destination HDF5 file path.
        """
        self.weather_dir = osp.join(weather_dir, 'PRSA_Data_20130301-20170228')
        self.output_path = output_path
        files = [f for f in os.listdir(self.weather_dir) if f.endswith(".csv")]
        self.df = pd.concat([pd.read_csv(osp.join(self.weather_dir, f)) for f in files])
        self.features = ['hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd_sin', 'wd_cos', 'WSPM']
        self.labels = ['station', 'year', 'month', 'day', 'season']

        def assign_season(month, day):
            if (month == 3 and day >= 15) or (4 <= month <= 5): return 'Spring'
            if 6 <= month <= 8: return 'Summer'
            if (9 <= month <= 10) or (month == 11 and day <= 15): return 'Autumn'
            return 'Winter'
        self.df['season'] = self.df[['month', 'day']].apply(lambda row: assign_season(row['month'], row['day']), axis=1)

        self.cls_mapping = {
            'station': {station: code for code, station in enumerate(self.df['station'].unique())},
            'year': {int(year): int(year - self.df['year'].min()) for year in self.df['year'].unique()},
            'month': {int(m): int(m - self.df['month'].min()) for m in self.df['month'].unique()},
            'day': {int(d): int(d - self.df['month'].min()) for d in self.df['day'].unique()},
            'season': {'Summer': 0, 'Autumn': 1, 'Winter': 2, 'Spring': 3},
        }
        for c in self.labels:
            self.df[c] = self.df[c].map(self.cls_mapping[c])

        self.wind_direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            np.nan: np.nan
        }
        self.df['wd_degrees'] = self.df['wd'].map(self.wind_direction_map)
        self.df['wd_radians'] = np.deg2rad(self.df['wd_degrees'])
        self.df['wd_sin'] = np.sin(self.df['wd_radians'])
        self.df['wd_cos'] = np.cos(self.df['wd_radians'])
        features_norm = [f for f in self.features if f not in ['hour', 'wd_sin', 'wd_cos']]
        means, stds = self.df[features_norm].mean(), self.df[features_norm].std()
        self.df[features_norm] = (self.df[features_norm] - means) / stds
        self.df[['RAIN', 'WSPM']] = self.df[['RAIN', 'WSPM']].fillna(0)


    def initialize(self, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15):
        """
        Processes and exports the structured air quality dataset.

        - Groups data by (station, year, month, day, season)
        - Interpolates missing values
        - Encodes features and labels
        - Writes data to HDF5 with train/val/test splits and class metadata

        :param val_size: Fraction of validation samples.
        :param test_size: Fraction of test samples.
        """
        groups = []
        labels = []
        for i, _g in self.df.groupby(self.labels):
            g = _g[self.features].sort_values(by='hour').drop(columns='hour')
            g = g.interpolate(method='linear', axis=0, limit_direction='both').fillna(0)
            groups.append(g.values)
            labels.append(i)
        data = np.stack(groups).astype(np.float32)
        labels = np.stack(labels)
        _idxs = np.arange(data.shape[0])
        rng = np.random.default_rng(seed)
        rng.shuffle(_idxs)
        data = data[_idxs]
        labels = labels[_idxs]

        classes = {s: {'index': i,
                       'type': 'static',
                       'n_classes': self.df[s].nunique(),
                       'ignore': False,
                       'values': self.cls_mapping[s]} for i, s in enumerate(self.labels)}

        N, T, _ = data.shape
        n1, n2 = int(N * (1 - val_size - test_size)), int(N * (1 - test_size))
        init_directories(osp.dirname(self.output_path))
        with h5py.File(self.output_path, 'w') as h5_file:
            h5_file.create_dataset('data', data=data, dtype=data.dtype)
            h5_file.create_dataset('labels', data=labels, dtype=labels.dtype)
            for set_, idxs in zip(['train', 'val', 'test'], [np.arange(n1), np.arange(n1, n2), np.arange(n2, N)]):
                h5_file.create_dataset(f'{set_}_indices', data=idxs)
            classes_json = json.dumps(classes)
            h5_file.attrs['classes'] = classes_json

if __name__ == '__main__':
    bmsaq_dir = r'/path/to/bms_air_quality'
    output_path = r'/path/to/create/bms_air_quality_dataset.h5'
    aq = BMSAirQualityPreprocess(bmsaq_dir, output_path)
    aq.initialize()
