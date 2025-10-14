from typing import Dict, List, Union

import numpy as np
from matplotlib import colors as mcolors

from msd.data.datasets.synthetic_video_generation.factor import CyclicSequence, HarmonicSequence, StaticFactor
from msd.data.datasets.synthetic_video_generation.factor_space import FactorSpace
from msd.data.datasets.synthetic_video_generation.state_mapper import StateMapper
from msd.data.datasets.synthetic_video_generation.video_generator import VideoGenerator


class dSpritesColored(StateMapper):
    """
    A class to handle the static dSprites dataset with colorization.
    It extends the StateMapper class and provides methods to retrieve colored images based on factor values.
    The class initializes with a dSprites file, color names, and metadata.
    """
    def __init__(self, dsprites_file: str, color_names: List[str] = ('DarkSlateGray', 'OrangeRed', 'Yellow', 'SpringGreen', 'Cyan', 'Purple', 'DodgerBlue', 'DeepPink', 'NavajoWhite')):
        """
        Initialize the dSpritesColored class with a dSprites file and color names.
        :param dsprites_file: Path to the dSprites dataset file.
        :param color_names: List of color names to be used for colorization.
        """
        data = np.load(dsprites_file, encoding="latin1", allow_pickle=True)
        self.metadata = data['metadata'][()]
        self.rgb_colors = {name: mcolors.to_rgb(name) for name in color_names}
        factor_names = self.metadata['latents_names']
        factor_values = {k: v.round(5) for k, v in self.metadata['latents_possible_values'].items()} | {'color': color_names, 'shape': ['square', 'ellipse', 'heart']}
        factors = [StaticFactor(n, factor_values[n]) for n in factor_names]
        images = data['imgs']
        labels = data['latents_classes']
        super().__init__(factors, images, labels)

    def get(self, factors: Dict[str, Union[int, str]]) -> np.ndarray:
        """
        Retrieve a colored image corresponding to given factor values.
        :param factors: A dictionary with keys as factor names and values as their corresponding values.
        :return: Corresponding RGB image of shape, dtype=uint8.
        """
        img = StateMapper.get(self, factors | {'color': 0})
        rgb = self.rgb_colors[self.factors['color'].values[factors['color']]]  # e.g. (1.0, 0.0, 0.0)
        colorized = np.stack([img * c for c in rgb]) * 255
        return colorized.astype(np.uint8)

class dSpritesStatic(VideoGenerator):
    """
    A class to generate static dSprites dataset.
    This class extends the VideoGenerator class and initializes it with a static dataset of dSprites.
    It creates a static dataset by applying various transformations to the static dataset.
    The transformations include changing the scale and orientation of the shapes.
    """
    def __init__(self, dsprites_file: str):
        ds = dSpritesColored(dsprites_file)
        factors = ds.factors
        static_factors = FactorSpace([
            factors['color'][:],
            factors['shape'][:],
            factors['posX'][2::4],
            factors['posY'][2::4],
        ])
        nScale, nOrient = len(factors['scale']), len(factors['orientation'])
        dynamic_factors = FactorSpace([
            factors['scale'].to_dynamic({
                'increasing_1x': HarmonicSequence(np.arange(0, nScale, 1)),
                'decreasing_1x': HarmonicSequence(np.arange(nScale-1, -1, -1)),
                'increasing_2x': HarmonicSequence(np.arange(0, nScale, 2)),
                'decreasing_2x': HarmonicSequence(np.arange(nScale-1, -1, -2)),
                'increasing_3x': HarmonicSequence(np.arange(0, nScale, nScale-1)),
                'decreasing_3x': HarmonicSequence(np.arange(nScale-1, -1, 1-nScale))
            }),
            factors['orientation'].to_dynamic({
                'counterclockwise': CyclicSequence(np.arange(0, nOrient, 2)),
                'static': CyclicSequence([0]),
                'clockwise': CyclicSequence(np.roll(np.arange(0, nOrient, 2)[::-1], 1)),
            })
        ])
        super().__init__(state_mapper=ds, static_factors=static_factors, dynamic_factors=dynamic_factors, T=16)

class dSpritesDynamic(VideoGenerator):
    """
    A class to generate dynamic dSprites dataset.
    This class extends the VideoGenerator class and initializes it with a static dataset of dSprites.
    It creates a dynamic dataset by applying various transformations to the static dataset.
    The transformations include changing the position of the shapes.
    """
    def __init__(self, dsprites_file: str):
        ds = dSpritesColored(dsprites_file)
        factors = ds.factors
        nO = len(factors['orientation'])
        static_factors = FactorSpace([
            factors['color'][:],
            factors['shape'][:],
            factors['scale'][:],
            factors['orientation'][np.arange(nO // 4)[::2]]
        ])

        nP = len(factors['posX'])
        dynamic_factors = FactorSpace([
            factors['posX'].to_dynamic({
                'right_1x': HarmonicSequence(np.arange(0, nP, 2)),
                'left_1x': HarmonicSequence(np.arange(nP-1, -1, -2)),
                'right_2x': HarmonicSequence(np.arange(0, nP, 4)),
                'left_2x': HarmonicSequence(np.arange(nP-1, -1, -4)),
                'mid_right': HarmonicSequence(np.arange(nP//2, nP, 1)),
                'mid_left': HarmonicSequence(np.arange(nP//2, -1, -1))
            }),
            factors['posY'].to_dynamic({
                'down_1x': HarmonicSequence(np.arange(0, nP, 2)),
                'up_1x': HarmonicSequence(np.arange(nP-1, -1, -2)),
                'down_2x': HarmonicSequence(np.arange(0, nP, 4)),
                'up_2x': HarmonicSequence(np.arange(nP-1, -1, -4)),
                'mid_down': HarmonicSequence(np.arange(nP//2, nP, 1)),
                'mid_up': HarmonicSequence(np.arange(nP//2, -1, -1))
            }),
        ])
        super().__init__(state_mapper=ds, static_factors=static_factors, dynamic_factors=dynamic_factors, T=12)


if __name__ == '__main__':
    data_path = r'/path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    dsprites_static = dSpritesStatic(data_path)
    dsprites_static.create_dataset('/path/to/create/dsprites_static_dataset.h5')
    dsprites_dynamic = dSpritesDynamic(data_path)
    dsprites_dynamic.create_dataset('/path/to/create/dsprites_dynamic_dataset.h5')
