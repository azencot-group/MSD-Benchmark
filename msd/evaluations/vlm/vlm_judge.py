from typing import Dict

from msd.evaluations.vlm.backbone.vlm_backbone import VLMBackbone
from msd.evaluations.vlm.prompt_setups.prompt_builder import VLMTaskBuilder, ContrastiveTaskBuilder
from msd.evaluations.vlm.prompt_setups.vlm_preprocessor import VLMPreprocessor


class VLMClassifierJudge:
    """
    A judge that performs classification using a Vision-Language Model (VLM) based on a single image or video clip.

    Supports both zero-shot and few-shot classification via prompt-based reasoning.
    """

    def __init__(self, vlm_backbone: VLMBackbone, preprocessor: VLMPreprocessor, prompt_builder: VLMTaskBuilder, examples=None):
        """
        :param vlm_backbone: An instance of VLMBackbone that provides generation and API calls.
        :param preprocessor: A VLMPreprocessor that prepares the image/video for VLM consumption.
        :param prompt_builder: A prompt builder that generates a user prompt for the classification task.
        :param examples: Optional tuple (X, Ys, Yd) for few-shot classification, where X is data, Ys and Yd are static & dynamic label dictionaries (respectively).
        """
        self.vlm_backbone = vlm_backbone
        self.preprocessor = preprocessor
        self.prompt_builder = prompt_builder
        self.examples = examples # Optional, for few-shot learning
        self.system_prompt = self.vlm_backbone.generate_system_prompt(self.preprocessor.system_prompt)
        self.examples_prompt = self.prepare_few_shot_examples(self.examples)


    def __call__(self, X):
        return self.classify(X)

    def classify(self, X):
        """
        Classify a list of video samples using the VLM.

        :param X: List of samples (e.g. sequences of frames).
        :return: List of JSON-structured predictions.
        """
        return [self._classify(x) for x in X]

    def _classify(self, x) -> Dict[str, str]:
        processed_data = self.preprocessor.process(x)
        user_prompt = self.vlm_backbone.generate_user_prompt(text=self.prompt_builder.user_prompt, image64=processed_data)
        prompt = [self.system_prompt] + self.examples_prompt + [user_prompt]
        response = self.vlm_backbone.generate_response(prompt)
        return response

    def prepare_few_shot_examples(self, examples):
        """
        Construct few-shot prompt examples from labeled data.

        :param examples: Tuple (X, Ys, Yd) of samples and their static/dynamic labels.
        :return: A list of prompts to be prepended before the main user prompt.
        """
        if examples is None:
            return []
        X, Ys, Yd = examples
        Y = Ys | Yd
        X = [self.preprocessor.process(x) for x in X]
        Y = [{k: Y[k][i] for k in Y.keys()} for i in range(len(X))]
        return self.vlm_backbone.generate_examples_prompt(X, Y, self.prompt_builder.user_prompt)

class VLMContrastiveJudge:
    """
    A contrastive judge that compares two video sequences and identifies visual differences or similarities.

    Designed for tasks with unknown or evolving feature spaces.
    """

    def __init__(self, vlm_backbone: VLMBackbone, preprocessor: VLMPreprocessor, prompt_builder: ContrastiveTaskBuilder, examples=None):
        """
        :param vlm_backbone: An instance of VLMBackbone that communicates with the VLM API.
        :param preprocessor: Preprocessor that encodes raw input data into base64 images.
        :param prompt_builder: Prompt generator for contrastive tasks.
        :param examples: Optional tuple (X1, X2, Y) of example image pairs and contrastive labels.
        """
        self.vlm_backbone = vlm_backbone
        self.preprocessor = preprocessor
        self.prompt_builder = prompt_builder
        self.examples = examples # Optional, for few-shot learning
        self.system_prompt = self.vlm_backbone.generate_system_prompt(self.preprocessor.system_prompt)
        self.examples_prompt = self.prepare_few_shot_examples(self.examples)

    def __call__(self, x1, x2):
        return self.compare(x1, x2)

    def compare(self, x1, x2):
        """
        Perform contrastive reasoning between two inputs.

        :param x1: First video sample.
        :param x2: Second video sample.
        :return: Dictionary describing differences and similarities between the inputs.
        """
        x1p = self.preprocessor.process(x1)
        x2p = self.preprocessor.process(x2)
        user_prompt = self.vlm_backbone.generate_user_prompt(text=self.prompt_builder.user_prompt, image64=[x1p, x2p])
        prompt = [self.system_prompt] + self.examples_prompt + [user_prompt]
        response = self.vlm_backbone.generate_response(prompt)
        return response

    def prepare_few_shot_examples(self, examples):
        """
        Construct few-shot examples for contrastive comparison.

        :param examples: Tuple (X1, X2, Y) where X1 and X2 are image pairs, and Y is the label dictionary.
        :return: List of message prompts to demonstrate few-shot tasks.
        """
        if examples is None:
            return []
        X1, X2, Y = examples
        X1 = [self.preprocessor.process(x1) for x1 in X1]
        X2 = [self.preprocessor.process(x2) for x2 in X2]
        X = [(x1, x2) for x1, x2 in zip(X1, X2)]
        Y = [{k: int(Y[k][i]) for k in Y.keys()} for i in range(len(X1))]
        return self.vlm_backbone.generate_examples_prompt(X, Y, self.prompt_builder.user_prompt)
