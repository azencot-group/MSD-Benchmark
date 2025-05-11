import json
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from msd.evaluations.vlm.backbone.vlm_backbone import VLMBackbone
from msd.evaluations.vlm.prompt_setups.prompt_builder import UnknownFeatureSpaceTaskBuilder, UnknownLabelSpaceTaskBuilder
from msd.evaluations.vlm.prompt_setups.vlm_preprocessor import DynamicVLMPreprocessor
from msd.evaluations.vlm.vlm_judge import VLMContrastiveJudge


class DatasetExplorer:
    """
    Uses a vision-language model (VLM) to explore and infer the visual factor structure of an unlabeled dataset.

    This class enables automatic discovery of:
    - Feature names that vary across samples.
    - Grouping of similar features into canonical factor names.
    - Classification of each factor as static or dynamic.
    - Estimation of label spaces for each factor.

    It leverages pairwise image comparison and natural language reasoning capabilities of a VLM.
    """

    def __init__(self, vlm: VLMBackbone, dataset: np.ndarray, num_pairs: int = 100, n_examples: int = 10):
        """
        :param vlm: A VLMBackbone-compatible model used for reasoning and analysis.
        :param dataset: A NumPy array of shape [N, T, C, H, W] or similar, where N is number of sequences.
        :param num_pairs: Number of pairs to use for contrastive analysis.
        :param n_examples: Number of sequences to include as prompt context examples.
        """
        self.vlm = vlm
        self.preprocessor = DynamicVLMPreprocessor()
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.n_examples = n_examples
        self.example_set = self.dataset[np.random.choice(self.dataset.shape[0], self.n_examples, replace=False)]
        self.example_decoded = [self.preprocessor.process(x) for x in self.example_set]

    def explore_factors(self, factors=None):
        """
        Compare image pairs using the VLM to extract visual features that differ between them.

        :param factors: List of factors to explore. If None, will attempt to discover factors automatically.
        :return: Dictionary mapping feature names to occurrence statistics and example differences.
        """
        judge = VLMContrastiveJudge(self.vlm,
                                    self.preprocessor,
                                    UnknownFeatureSpaceTaskBuilder() if factors is None else UnknownLabelSpaceTaskBuilder(factors),
                                    examples=None)
        indices = np.random.choice(self.dataset.shape[0], self.num_pairs * 2, replace=False)
        features = {}
        for i in tqdm(range(self.num_pairs)):
            x1, x2 = self.dataset[indices[i]], self.dataset[indices[i + self.num_pairs]]
            result = judge(x1, x2)
            if "different" in result and isinstance(result["different"], dict):
                for key in result["different"].keys():
                    d = features.get(key, {})
                    d["count"] = d.get("count", 0) + 1
                    diffs = d.get("different", [])
                    diffs.append(result["different"][key])
                    d["different"] = diffs
                    features[key] = d
        return features

    def query(self, text, examples=None):
        """
        Submit a prompt to the VLM along with base64-encoded examples.

        :param text: Instruction or question string.
        :param examples: List of base64-encoded images as examples.
        :return: VLM-generated JSON-parsed response.
        """
        prompt = [self.vlm.generate_user_prompt(text=text, image64=examples)]
        result = self.vlm.generate_response(prompt)
        return result

    def find_unique(self, features, min_repeats):
        """
        Group repeated features into canonical names using VLM summarization.

        :param features: Dictionary of features and their example counts.
        :param min_repeats: Minimum number of times a feature must appear to be considered.
        :return: Dictionary mapping canonical factor names to a list of raw feature descriptions.
        """
        text = (
            "You are given a dictionary of visual feature names and their frequency counts, extracted from image comparisons.\n"
            "Many features describe the same concept using different wording.\n\n"
            f"Your task is to group only recurring, meaningful features (those with a count ≥ {min_repeats}) into clusters.\n"
            "Each cluster should represent a distinct (object + attribute) pair like 'shirt color', 'pants color', or 'pose type'.\n\n"
            f"Ignore all features with a count less than {min_repeats}.\n\n"
            f"Here is the feature list:\n{json.dumps(features, indent=2)}\n\n"
            "Return a JSON dictionary where each key is a canonical factor (e.g. 'shirt color') and the value is a list of grouped feature names."
        )
        return self.query(text, self.example_decoded)

    def classify_type(self, features: list[str]) -> dict:
        """
        Classify visual features as either 'static' or 'dynamic'.

        :param features: List of canonical feature names.
        :return: Dictionary mapping each feature to 'static' or 'dynamic'.
        """
        features_text = ", ".join(features)
        text = (
            "You are given a list of visual features extracted from video sequences of people. "
            "Your task is to classify whether each feature is typically static (unchanging across time in a given sequence) "
            "or dynamic (changes over the course of a sequence).\n\n"
            f"The features to classify are:\n{features_text}\n\n"
            "Below are some example frames sampled from the dataset to help you understand the visual context. "
            "These examples may include variations in clothing, pose, motion, and background.\n\n"
            "Please respond with a JSON dictionary where each key is a feature name and each value is either 'static' or 'dynamic'."
        )
        return self.query(text, self.example_decoded)

    def explore_labels(self, unique_factors: List[str], differences: Dict[str, List[str]]) -> dict:
        """
        Estimate the label space for each canonical factor using natural-language evidence.

        :param unique_factors: Canonical feature names.
        :param differences: Dictionary mapping each feature to observed descriptions.
        :return: Dictionary mapping each feature to a list of predicted label values.
        """
        factors_text = "\n".join(f"- {f}" for f in unique_factors)
        observed_text = json.dumps(differences, indent=2)

        text = (
            "You are given a list of visual features (e.g., 'shirt color', 'pose type') and two sources of evidence:\n"
            "1. Natural-language descriptions of how each feature varies across image pairs.\n"
            "2. A set of visual examples showing those differences (provided separately).\n\n"
            "Your task is to estimate the most likely set of **distinct, meaningful label values** for each feature.\n\n"
            "**Instructions:**\n"
            "- Consider both the textual differences and the visual evidence.\n"
            "- Only include label values that are visually and semantically distinct.\n"
            "- **Group similar values together** (e.g., 'light blue' and 'blue' → 'blue').\n"
            "- **Avoid rare, overly specific labels**.\n"
            "- For each feature, **keep the list short and practical** (ideally ≤ 8 values).\n"
            "- The goal is to define a usable label space for each feature in a classification task.\n\n"
            f"**Features:**\n{factors_text}\n\n"
            f"**Observed Differences (textual):**\n{observed_text}\n\n"
            "Return your answer as a **JSON dictionary**.\n"
            "Each key should be a feature name, and each value should be a list of typical label values.\n"
        )

        return self.query(text, self.example_decoded)
