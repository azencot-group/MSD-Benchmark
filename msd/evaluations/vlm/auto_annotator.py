import json
from os import path as osp
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

from msd.evaluations.vlm.backbone.vlm_backbone import VLMBackbone
from msd.evaluations.vlm.dataset_explorer import DatasetExplorer
from msd.evaluations.vlm.prompt_setups.prompt_builder import ClosedLabelTaskBuilder
from msd.evaluations.vlm.prompt_setups.vlm_preprocessor import VLMPreprocessor
from msd.evaluations.vlm.vlm_judge import VLMClassifierJudge
from msd.utils.loading_utils import read_json, write_json


class AutoAnnotator:
    """
    Automatically annotates a video dataset using a Vision-Language Model (VLM) and structured reasoning prompts.

    The annotation pipeline consists of two phases:
    1. Feature discovery and class definition (via contrastive comparisons).
    2. Sample-by-sample labeling of dynamic video segments.
    """

    def __init__(self, vlm: VLMBackbone, preprocessor: VLMPreprocessor, out_path: str):
        """
        :param vlm: A VLMBackbone-compatible model for classification and comparison.
        :param preprocessor: A VLMPreprocessor for encoding video sequences.
        :param out_path: Directory where annotation results and metadata will be saved.
        """
        self.vlm = vlm
        self.preprocessor = preprocessor
        self.out_path = out_path

    def explore_dataset(self, exploration_set: np.ndarray, dataset_name: str, num_pairs: int=None, factors: List[str]=None) -> Dict:
        """
        Discover visual factors and label spaces from a training set using VLM-guided analysis.

        Computes:
        - Frequently observed visual differences.
        - Canonical factor names.
        - Factor type (static/dynamic).
        - Label values for each factor.

        :param exploration_set: A NumPy array of video samples for exploration.
        :param dataset_name: Unique name used for caching results.
        :param num_pairs: Number of pairs used for exploration.
        :param factors: Optional list of known factors for label-space exploration. If None, will attempt to discover factors automatically.
        :return: Dictionary of factor metadata suitable for classification.
        """
        if num_pairs is None:
            num_pairs = len(exploration_set) // 2
        n_examples = num_pairs // 10
        min_repeats = n_examples // 2
        explorer = DatasetExplorer(self.vlm, exploration_set, num_pairs=num_pairs, n_examples=n_examples)
        classes_file = osp.join(self.out_path, f'{dataset_name}_classes.json')
        if osp.exists(classes_file):
            classes = read_json(classes_file)
        else:
            exploration_file = osp.join(self.out_path, f'{dataset_name}_exploration.json')
            if osp.exists(exploration_file):
                exploration = read_json(exploration_file)
            else:
                exploration = explorer.explore_factors(factors=factors)
                write_json(exploration_file, exploration)
            factors_counts = {k: v['count'] for k, v in exploration.items()}
            unique_factors = explorer.find_unique(factors_counts, min_repeats)
            unique_factors = explorer.classify_type(unique_factors)
            differences_descriptions = {k: v['different'] for k, v in exploration.items()}
            label_space = explorer.explore_labels(list(unique_factors.keys()), differences_descriptions)

            classes = {}
            for i, k in enumerate(unique_factors.keys()):
                cls = {
                    'index': i,
                    'type': unique_factors[k],
                    'n_classes': len(label_space[k]),
                    'ignore': False,
                    'values': {v: i for i, v in enumerate(label_space[k])},
                }
                classes[k] = cls
            write_json(classes_file, classes)
        return classes

    def annotate_dataset(self, annotation_set: np.ndarray, dataset_name: str, classes: Dict, org_indices: np.ndarray, example_set=None) -> list:
        """
        Annotate a dataset by predicting structured visual factors for each sequence using the VLM.

        :param annotation_set: Array of video samples to annotate.
        :param dataset_name: Name used to store annotated data.
        :param classes: Metadata about discovered factors.
        :param org_indices: Original indices of the samples (used to track source).
        :param example_set: Optional base64-encoded context examples used for few-shot learning to provide the model with context for better predictions.
        :return: A list of label vectors corresponding to each video sample.
        """
        label_space = {k: [c for c in v['values'].keys()] for k, v in classes.items()}
        judge = VLMClassifierJudge(self.vlm, self.preprocessor, ClosedLabelTaskBuilder(label_space), example_set)
        sorted_factors = sorted(classes.items(), key=lambda x: x[1]['index'])
        factors = [name for name, _ in sorted_factors]
        Y = []
        for i, x in tqdm(enumerate(annotation_set)):
            y = judge([x])[0]
            _y = []
            for f in factors:
                try:
                    _y.append(classes[f]['values'][y[f]])
                except KeyError:
                    print(f'Error in {f} for {y}')
                    _y.append(-1)
            Y.append(_y)
        X = (annotation_set * 255).astype(np.uint8)
        test_indices = np.arange(X.shape[0])
        with h5py.File(osp.join(self.out_path, f'{dataset_name}.h5'), 'w') as h5_file:
            h5_file.create_dataset('data', data=X, dtype=X.dtype)
            h5_file.create_dataset('labels', data=np.array(Y))
            h5_file.create_dataset('test_indices', data=test_indices)
            h5_file.create_dataset('org_indices', data=org_indices)
            h5_file.attrs['classes'] = json.dumps(classes)
        return Y
