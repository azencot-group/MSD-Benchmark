from itertools import combinations

import numpy as np

RAW_DATA_FOLDER = '.'
DATASETS_FOLDER = 'data'
LATENT_INFO_CSV = 'dMelodies_dataset_latent_info.csv'
NPZ_DATASET = 'dMelodies_dataset.npz'

CHORD_DICT = {'I': [0, 2, 4],
              'IV': [3, 5, 7]}

TONIC_LIST = ['C',  'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
TONIC_DICT = {i: t for i, t in enumerate(TONIC_LIST)}
TONIC_REVERSE_DICT = {TONIC_DICT[k]: k for k in TONIC_DICT.keys()}

OCTAVE_LIST = [4]
OCTAVE_DICT = {i: o for i, o in enumerate(OCTAVE_LIST)}
OCTAVE_REVERSE_DICT = {OCTAVE_DICT[k]: k for k in OCTAVE_DICT.keys()}

SCALE_LIST = ['major', 'minor', 'blues']
SCALE_DICT = {i: m for i, m in enumerate(SCALE_LIST)}
SCALE_REVERSE_DICT = {SCALE_DICT[k]: k for k in SCALE_DICT.keys()}
SCALE_NOTES_DICT = {'major': [0, 2, 4, 5, 7, 9, 11, 12, 14],
                    'minor': [0, 2, 3, 5, 7, 8, 11, 12, 14],
                    'blues': [0, 2, 3, 5, 6, 9, 10, 12, 14]}

RHYTHM_DICT = {}
all_rhythms = combinations([0, 1, 2, 3, 4, 5, 6, 7], 6)
for i, pos in enumerate(list(all_rhythms)):
    temp_array = np.array([0] * 8)
    temp_array[np.array(pos)] = 1
    RHYTHM_DICT[i] = list(temp_array)

ARP_DICT = {0: 'up',
            1: 'down'}
ARP_REVERSE_DICT = {ARP_DICT[k]: k for k in ARP_DICT.keys()}
