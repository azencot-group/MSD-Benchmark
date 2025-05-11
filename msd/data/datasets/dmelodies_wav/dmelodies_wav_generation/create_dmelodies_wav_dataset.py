import os
import warnings

from midi_ddsp import load_pretrained_model
from midi_ddsp.data_handling.instrument_name_utils import INST_NAME_TO_ID_DICT
from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi
from midi_ddsp.utils.training_utils import set_seed

warnings.filterwarnings('ignore')

set_seed(1234)

synthesis_generator, expression_generator = load_pretrained_model()

for file in os.listdir('./midi'):
    midi_file = f'./midi/{file}'
    for instrument_name in ('violin', 'trumpet', 'saxophone', 'flute'):
        instrument_id = INST_NAME_TO_ID_DICT[instrument_name]
        output_dir = f'./wav/{instrument_name}'
        os.makedirs(output_dir, exist_ok=True)
        pitch_offset = 0
        speed_rate = 1

        synthesize_mono_midi(synthesis_generator, expression_generator, midi_file, instrument_id,
                             output_dir=output_dir, pitch_offset=pitch_offset, speed_rate=speed_rate)
