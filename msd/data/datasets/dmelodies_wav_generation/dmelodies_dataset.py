import os
from typing import Union

import music21
import numpy as np
from constants import (ARP_DICT, ARP_REVERSE_DICT, DATASETS_FOLDER, NPZ_DATASET, OCTAVE_DICT, OCTAVE_REVERSE_DICT,
                       RHYTHM_DICT, SCALE_DICT, SCALE_REVERSE_DICT, TONIC_DICT, TONIC_REVERSE_DICT)
from helpers import (SLUR_SYMBOL, TICK_VALUES, compute_tick_durations, get_latent_info, get_notes,
                     get_score_for_item, is_score_on_ticks, standard_name)
from tqdm import tqdm


class DMelodiesDataset:
    def __init__(self, num_data_points: int = None):
        self.df = get_latent_info()
        if num_data_points is None:
            self.num_data_points = self.df.shape[0]
        else:
            self.num_data_points = num_data_points
        self.time_sig_num = 4
        self.time_sig_den = 4
        self.beat_subdivisions = len(TICK_VALUES)
        self.tick_durations = compute_tick_durations(TICK_VALUES)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(os.path.join(cur_dir, DATASETS_FOLDER)):
            os.mkdir(os.path.join(cur_dir, DATASETS_FOLDER))
        self.dataset_path = os.path.join(cur_dir, DATASETS_FOLDER, NPZ_DATASET)
        self.score_array = None
        self.latent_array = None
        self.metadata = None
        self.latent_dicts = {'tonic': TONIC_DICT,
                             'octave': OCTAVE_DICT,
                             'scale': SCALE_DICT,
                             'rhythm_bar1': RHYTHM_DICT,
                             'arp_chord1': ARP_DICT,
                             'arp_chord2': ARP_DICT}
        self.note2index_dict = dict()
        self.index2note_dict = dict()
        self.initialize_index_dicts()

    def _get_score_for_item(self, index: int) -> music21.stream.Score:
        assert 0 <= index < self.num_data_points
        d = self.df.iloc[index]
        return get_score_for_item(d)

    def make_or_load_dataset(self):
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            dataset = np.load(self.dataset_path, allow_pickle=True)
            self.score_array = dataset['score_array']
            self.latent_array = dataset['latent_array']
            self.note2index_dict = dataset['note2index_dict'].item()
            self.index2note_dict = dataset['index2note_dict'].item()
            self.latent_dicts = dataset['latent_dicts'].item()
            self.metadata = dataset['metadata'].item()
            return

        print('Making tensor dataset')
        score_seq = [None] * self.num_data_points
        latent_seq = [None] * self.num_data_points

        def _create_data_point(item_index):
            m21_score = self._get_score_for_item(item_index)
            score_array = self.get_tensor(m21_score)
            score_seq[item_index] = score_array
            latent_array = self._get_latents_array_for_index(item_index)
            latent_seq[item_index] = latent_array

        for idx in tqdm(range(self.num_data_points)):
            _create_data_point(idx)

        self.score_array = np.array(score_seq)
        self.latent_array = np.array(latent_seq)
        print('Number of data points: ', self.score_array.shape[0])
        # noinspection PyTypeChecker
        np.savez(self.dataset_path,
                 score_array=self.score_array,
                 latent_array=self.latent_array,
                 note2index_dict=self.note2index_dict,
                 index2note_dict=self.index2note_dict,
                 latent_dicts=self.latent_dicts)

    def get_tensor(self, score: music21.stream.Score) -> Union[np.array, None]:
        eps = 1e-5
        notes = get_notes(score)
        if not is_score_on_ticks(score, TICK_VALUES):
            return None
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi) for n in notes if n.isNote]
        for note_name, pitch in list_note_strings_and_pitches:

            if note_name not in self.note2index_dict:
                self.update_index_dicts(note_name)

        x = 0
        y = 0
        length = int(score.highestTime * self.beat_subdivisions)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0
        while y < length:
            if x < num_notes - 1:
                if notes[x + 1].offset > current_tick + eps:
                    t[y, :] = [self.note2index_dict[standard_name(notes[x])], is_articulated]
                    y += 1
                    current_tick += self.tick_durations[(y - 1) % len(TICK_VALUES)]
                    is_articulated = False
                else:
                    x += 1
                    is_articulated = True
            else:
                t[y, :] = [self.note2index_dict[standard_name(notes[x])], is_articulated]
                y += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * self.note2index_dict[SLUR_SYMBOL]
        lead = lead.astype('int32')
        return lead

    def _get_latents_array_for_index(self, index: int) -> np.array:
        assert 0 <= index < self.num_data_points
        d = self.df.iloc[index]
        latent_list = [TONIC_REVERSE_DICT[d['tonic']],
                       OCTAVE_REVERSE_DICT[d['octave']],
                       SCALE_REVERSE_DICT[d['scale']],
                       d['rhythm_bar1'],
                       ARP_REVERSE_DICT[d['arp_chord1']],
                       ARP_REVERSE_DICT[d['arp_chord2']]]
        return np.array(latent_list).astype('int32')

    def initialize_index_dicts(self):
        note_sets = set()
        note_sets.add(SLUR_SYMBOL)
        note_sets.add('rest')
        for note_index, note_name in enumerate(note_sets):
            self.index2note_dict.update({note_index: note_name})
            self.note2index_dict.update({note_name: note_index})

    def update_index_dicts(self, new_note_name):
        new_index = len(self.note2index_dict)
        self.index2note_dict.update({new_index: new_note_name})
        self.note2index_dict.update({new_note_name: new_index})
        print(f'Warning: Entry {str({new_index: new_note_name})} added to dictionaries')
