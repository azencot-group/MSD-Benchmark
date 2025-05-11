import os
from fractions import Fraction
from itertools import product
from typing import Union

import music21
import pandas as pd
from tqdm import tqdm

from constants import ARP_DICT, CHORD_DICT, LATENT_INFO_CSV, OCTAVE_DICT, RHYTHM_DICT, SCALE_DICT, SCALE_NOTES_DICT, TONIC_DICT

SLUR_SYMBOL = '__'
TICK_VALUES = [0, Fraction(1, 2)]


def create_latent_info_df() -> pd.DataFrame:
    tonic_list = []
    octave_list = []
    scale_list = []
    rhy1_list = []
    dir1_list = []
    dir2_list = []

    all_combinations = product(TONIC_DICT.keys(),
                               OCTAVE_DICT.keys(),
                               SCALE_DICT.keys(),
                               RHYTHM_DICT.keys(),
                               ARP_DICT.keys(),
                               ARP_DICT.keys())
    for params in tqdm(all_combinations):
        tonic_list.append(TONIC_DICT[params[0]])
        octave_list.append(OCTAVE_DICT[params[1]])
        scale_list.append(SCALE_DICT[params[2]])
        rhy1_list.append(params[3])
        dir1_list.append(ARP_DICT[params[4]])
        dir2_list.append(ARP_DICT[params[5]])
    d = {'tonic': tonic_list,
         'octave': octave_list,
         'scale': scale_list,
         'rhythm_bar1': rhy1_list,
         'arp_chord1': dir1_list,
         'arp_chord2': dir2_list}
    latent_df = pd.DataFrame(data=d)
    return latent_df


def get_latent_info() -> pd.DataFrame:
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    latent_info_path = os.path.join(cur_dir, LATENT_INFO_CSV)
    if os.path.exists(latent_info_path):
        # noinspection PyTypeChecker
        latent_df = pd.read_csv(latent_info_path, index_col=0)
    else:
        latent_df = create_latent_info_df()
        # noinspection PyTypeChecker
        latent_df.to_csv(path_or_buf=latent_info_path)
    return latent_df


def get_midi_pitch_list(tonic: str,
                        octave: int,
                        mode: str,
                        arp_dir1: str,
                        arp_dir2: str) -> list:
    root_pitch = music21.pitch.Pitch(tonic + str(octave)).midi
    pitch_seq = []
    dir_seq = [arp_dir1, arp_dir2]
    for index, chord in enumerate(CHORD_DICT.keys()):
        seq = CHORD_DICT[chord]
        if dir_seq[index] == 'down':
            seq = seq[::-1]
        for s in seq:
            midi_pitch = root_pitch + SCALE_NOTES_DICT[mode][s]
            pitch_seq.append(midi_pitch)
    return pitch_seq


def create_m21_melody(tonic: str,
                      octave: int,
                      mode: str,
                      rhythm_bar1: int,
                      arp_dir1: str,
                      arp_dir2: str) -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    dur = 0.0
    rhy1 = RHYTHM_DICT[rhythm_bar1]
    if sum(rhy1) != 6:
        raise (ValueError, f'Invalid rhythm: {rhy1}')
    midi_pitch_seq = get_midi_pitch_list(tonic, octave, mode, arp_dir1, arp_dir2)
    curr_note_num = 0
    for rhy in [rhy1]:
        for onset in rhy:
            if onset == 1:
                f = music21.note.Note()
                f.pitch.midi = midi_pitch_seq[curr_note_num]
                f.duration = music21.duration.Duration('eighth')
                curr_note_num += 1
            else:
                f = music21.note.Rest()
                f.duration = music21.duration.Duration('eighth')
            part.insert(dur, f)
            dur += music21.duration.Duration('eighth').quarterLength

    score.insert(part)
    return score


def get_score_for_item(df_row: pd.Series) -> music21.stream.Score:
    return create_m21_melody(tonic=df_row['tonic'],
                             octave=df_row['octave'],
                             mode=df_row['scale'],
                             rhythm_bar1=df_row['rhythm_bar1'],
                             arp_dir1=df_row['arp_chord1'],
                             arp_dir2=df_row['arp_chord2'])


def get_file_name_for_item(df_row: pd.Series, index: int) -> str:
    tonic = df_row['tonic']
    octave = df_row['octave']
    mode = df_row['scale']
    rhythm_bar1 = df_row['rhythm_bar1']
    dir1 = df_row['arp_chord1']
    dir2 = df_row['arp_chord2']
    file_name = f'{index}_{tonic}_{octave}_{mode}_{rhythm_bar1}_{dir1}_{dir2}'
    return file_name


def compute_tick_durations(tick_values: list):
    diff = [n - p for n, p in zip(tick_values[1:], tick_values[:-1])]
    diff = diff + [1 - tick_values[-1]]
    return diff


def get_notes(score: music21.stream.Score) -> list:
    notes = score.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    return notes


def is_score_on_ticks(score: music21.stream.Score, tick_values: list) -> bool:
    notes = get_notes(score)
    eps = 1e-5
    for n in notes:
        _, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_values:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False
    return True


def standard_name(note_or_rest: Union[music21.note.Note, music21.note.Rest]) -> str:
    if isinstance(note_or_rest, music21.note.Note):
        return note_or_rest.nameWithOctave
    elif isinstance(note_or_rest, music21.note.Rest):
        return note_or_rest.name
    else:
        raise ValueError('Invalid input. Should be a music21.note.Note or music21.note.Rest object ')
