from argparse import ArgumentParser

import numpy as np

from msd.data.hooks import ToNumpy, Transpose, Normalize
from msd.data.readers.h5_reader import Hdf5Reader
from msd.data.disentanglement_dataset import DisentanglementDataset
from msd.evaluations.vlm.auto_annotator import AutoAnnotator
from msd.evaluations.vlm.backbone.openai_backbone import OpenAIGPTBackbone
from msd.evaluations.vlm.backbone.qwen25_backbone import QwenVLBackbone
from msd.evaluations.vlm.prompt_setups.vlm_preprocessor import DynamicVLMPreprocessor
from msd.utils.loading_utils import read_json

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--ds_path', type=str, help='Path to a DisentanglementDataset .h5 file')
    parser.add_argument('--exploration_subset', type=str, choices=['train', 'val', 'test'], help='Subset to load for exploration')
    parser.add_argument('--annotation_subset', type=str, choices=['train', 'val', 'test'], help='Subset to load for annotation')

    parser.add_argument('--vlm_backbone', type=str, choices=['openai', 'qwen'], default='qwen')

    parser.add_argument('--n_exploration', type=int, default=None, help='Number of samples to load. If not specified, will use the rest of the dataset for exploration')
    parser.add_argument('--n_annotation', type=int, default=None, help='Number of samples to load. If not specified, will use the rest of the dataset')

    parser.add_argument('--out_dir', type=str, help='Path to save the annotated dataset')
    parser.add_argument('--ds_name', type=str, help='Name of the dataset to save')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    def get_subset(subset, n=None):
        reader = Hdf5Reader(args.ds_path, split=subset)
        ds = DisentanglementDataset(reader, preprocess_hooks=[Normalize(0, 1)], supervised=False)
        if n is None:
            n = len(ds)
        indices = rng.choice(len(ds), n, replace=False)[:]
        return ds[indices], indices

    exploration_set, _ = get_subset(args.exploration_subset, args.n_exploration)
    annotation_set, ann_indices = get_subset(args.annotation_subset, args.n_annotation)

    if args.vlm_backbone == 'openai':
        # Use OpenAI GPT-4o as the VLM backbone
        vlm = OpenAIGPTBackbone()
    elif args.vlm_backbone == 'qwen':
        # Use Qwen-2.5 as the VLM backbone
        vlm = QwenVLBackbone()
    else:
        raise ValueError(f"Unsupported VLM type: {args.vlm_type}")
    preprocessor = DynamicVLMPreprocessor()
    ann = AutoAnnotator(vlm, preprocessor, out_path=args.out_dir)
    feature_space = ann.explore_dataset(exploration_set, args.ds_name)
    ann.annotate_dataset(annotation_set, args.ds_name, feature_space, org_indices=ann_indices)