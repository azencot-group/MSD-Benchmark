from argparse import ArgumentParser

import numpy as np

from msd.data.readers.h5_reader import Hdf5Reader
from msd.data.disentanglement_dataset import DisentanglementDataset
from msd.evaluations.vlm.auto_annotator import AutoAnnotator
from msd.evaluations.vlm.backbone.gpt_judge import OpenAIGPTBackbone
from msd.evaluations.vlm.setups.vlm_preprocessor import DynamicVLMPreprocessor

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--ds_path', type=str, help='Path to a DisentanglementDataset .h5 file')
    parser.add_argument('--subset', type=str, choices=['train', 'val', 'test'], help='Subset to load')

    parser.add_argument('--n_exploration', type=int, default=500, help='Number of samples to load')
    parser.add_argument('--n_annotation', type=int, default=500, help='Number of samples to load')

    parser.add_argument('--out_dir', type=str, help='Path to save the annotated dataset')
    parser.add_argument('--ds_name', type=str, help='Name of the dataset to save')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reader = Hdf5Reader(args.ds_path, split=args.subset)
    ds = DisentanglementDataset(reader, supervised=False)
    indices = rng.choice(len(ds), args.n_exploration + args.n_annotation, replace=False)
    ann_indices = indices[args.n_exploration:]
    exploration_set = ds[indices[:args.n_exploration]]
    annotation_set = ds[ann_indices]

    vlm = OpenAIGPTBackbone()
    preprocessor = DynamicVLMPreprocessor()
    ann = AutoAnnotator(vlm, preprocessor, out_path=args.out_dir)
    feature_space = ann.explore_dataset(exploration_set, args.ds_name)
    ann.annotate_dataset(annotation_set, args.ds_name, feature_space, org_indices=ann_indices)