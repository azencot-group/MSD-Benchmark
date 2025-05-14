import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

from msd.data.datasets.synthetic_video_generation.factor import StaticFactor, HarmonicSequence, CyclicSequence
from msd.data.datasets.synthetic_video_generation.factor_space import FactorSpace
from msd.data.datasets.synthetic_video_generation.state_mapper import StateMapper
from msd.data.datasets.synthetic_video_generation.video_generator import VideoGenerator


class Shapes3D(StateMapper):
    def __init__(self, dataset_file: str):
        """
        Initialize the static 3DShapes dataset with a file path.
        :param dataset_file: Path to the dataset file.
        """
        with h5py.File(dataset_file, 'r') as dataset:
            images = np.array(dataset['images']).transpose((0, 3, 1, 2))  # (N, C, H, W)
            values = np.array(dataset['labels']).round(5)
        factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        factor_values = {k: sorted(np.unique(values[:, i])) for i, k in enumerate(factor_names)} | {'shape': ['cube', 'cylinder', 'sphere', 'capsule']}
        factors = [StaticFactor(n, factor_values[n]) for n in factor_names]
        labels = np.zeros_like(values, dtype=int)
        for col in range(values.shape[1]):
            encoder = LabelEncoder()
            labels[:, col] = encoder.fit_transform(values[:, col])

        super().__init__(factors, images, labels)

class Dynamic3DShapes(VideoGenerator):
    """
    A class to generate dynamic 3D shapes dataset.
    This class extends the VideoGenerator class and initializes it with a static dataset of 3D shapes.
    It creates a dynamic dataset by applying various transformations to the static dataset.
    The transformations include changing the scale and orientation of the shapes.
    """
    def __init__(self, dataset_file: str):
        ds = Shapes3D(dataset_file)
        factors = ds.factors
        static_factors = FactorSpace([
            factors['floor_hue'][:],
            factors['wall_hue'][:],
            factors['object_hue'][:],
            factors['shape'][:],
        ])
        nScale, nOrient = len(factors['scale']), len(factors['orientation'])
        dynamic_factors = FactorSpace([
            factors['scale'].to_dynamic({
                'increasing_1x': HarmonicSequence(np.arange(0, nScale, 1)),
                'decreasing_1x': HarmonicSequence(np.arange(nScale-1, -1, -1)),
                'increasing_2x': HarmonicSequence(np.arange(0, nScale, 2)),
                'decreasing_2x': HarmonicSequence(np.arange(nScale-1, -1, -2)),
                'alternating_big_small': HarmonicSequence(np.arange(0, nScale, nScale-1)),
                'alternating_small_big': HarmonicSequence(np.arange(nScale-1, -1, 1-nScale))
            }),
            factors['orientation'].to_dynamic({
                'counterclockwise': CyclicSequence(np.arange(0, nOrient, 2)),
                'static': CyclicSequence([0]),
                'clockwise': CyclicSequence(np.roll(np.arange(0, nOrient, 2)[::-1], 1)),
            })
        ])
        super().__init__(state_mapper=ds, static_factors=static_factors, dynamic_factors=dynamic_factors, T=10)


if __name__ == '__main__':
    static_dataset_file = '/path/to/3dshapes.h5'
    output_file = '/path/to/3dshapes_dataset.h5'

    dynamic3D_shapes = Dynamic3DShapes(static_dataset_file)
    dynamic3D_shapes.create_dataset(out_path=output_file)