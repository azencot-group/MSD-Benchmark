import argparse
import multiprocessing
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from constants import RAW_DATA_FOLDER
from dmelodies_dataset import DMelodiesDataset
from helpers import get_file_name_for_item, get_score_for_item


def save_(index: int,
          data_row: pd.Series,
          save_mid: bool = False,
          save_xml: bool = False):
    if not (save_mid or save_xml):
        return
    score = get_score_for_item(data_row)
    file_name = get_file_name_for_item(data_row, index)
    if save_mid:
        midi_save_path = os.path.join(RAW_DATA_FOLDER, 'midi', file_name + '.mid')
        if not os.path.exists(os.path.dirname(midi_save_path)):
            os.makedirs(os.path.dirname(midi_save_path))
        score.write('midi', midi_save_path)
    if save_xml:
        xml_save_path = os.path.join(RAW_DATA_FOLDER, 'musicxml', file_name + '.musicxml')
        if not os.path.exists(os.path.dirname(xml_save_path)):
            os.makedirs(os.path.dirname(xml_save_path))
        score.write('musicxml', xml_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-mid', help='save data points in .mid format (default: false', action='store_true')
    parser.add_argument('--save-xml', help='save data points in .mid format (default: false', action='store_true')
    parser.add_argument('--debug', help='flag to create a smaller subset for debugging', action='store_true')
    args = parser.parse_args()
    s_mid = args.save_mid
    s_xml = args.save_xml
    debug = args.debug

    num_data_points = None
    if debug:
        num_data_points = 1000
    dataset = DMelodiesDataset(num_data_points=num_data_points)
    dataset.make_or_load_dataset()

    df = dataset.df.head(n=dataset.num_data_points)
    if debug:
        for i, d in tqdm(df.iterrows()):
            save_(i, d, s_mid, s_xml)
    else:
        cpu_count = multiprocessing.cpu_count()
        print(cpu_count)
        Parallel(n_jobs=cpu_count)(delayed(save_)(i, d, s_mid, s_xml) for i, d in tqdm(df.iterrows()))
