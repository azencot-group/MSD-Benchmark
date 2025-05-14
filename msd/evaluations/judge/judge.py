from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch import nn

from msd.configurations.msd_component import MSDComponent
from msd.evaluations.vlm.prompt_setups.prompt_builder import ClosedLabelTaskBuilder
from msd.evaluations.vlm.prompt_setups.vlm_preprocessor import DynamicVLMPreprocessor
from msd.evaluations.vlm.vlm_judge import VLMClassifierJudge

if TYPE_CHECKING:
    from msd.configurations.config_initializer import ConfigInitializer

class Judge(MSDComponent):
    """
    Base class marker for all judge implementations.
    """
    pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class ClassifierJudge(nn.Module, Judge):
    """
    Torch-based judge that uses a pretrained classifier to produce predictions.
    Typically used in fully supervised setups with model checkpoints.
    """

    def __init__(self, initializer: 'ConfigInitializer', classifier_cfg: DictConfig, classifier_loader_cfg: DictConfig):
        """
        :param initializer: ConfigInitializer used to build the classifier module.
        :param classifier_cfg: Configuration used to construct the classifier.
        :param classifier_loader_cfg: Configuration used to construct the classifier loader.
        """
        super(ClassifierJudge, self).__init__()
        self.loader = initializer.initialize(classifier_loader_cfg)
        self.checkpoint = self.loader.load_classifier()
        args = self.checkpoint['arguments']
        args['classes'] = self.checkpoint['classes']
        self.classifier = initializer.initialize(classifier_cfg, **args)
        self.classifier.load_state_dict(self.checkpoint['model'])

    def forward(self, X, frame_level: bool = False):
        """
        Perform forward prediction using the classifier.

        :param X: Input tensor.
        :param frame_level: Whether to return frame-level predictions or sequence-level.
        :return: Dictionary of predicted class indices per factor.
        """
        predictions = self.classifier.classify(X, frame_level)
        return predictions

class VLMJudge(Judge):
    """
    A Vision-Language Model (VLM)-based judge that performs classification using LLM reasoning over image inputs.

    This judge is particularly useful for zero-shot and few-shot evaluation setups, where manual training is not available.
    """

    def __init__(self, initializer: 'ConfigInitializer', vlm_backbone_cfg: DictConfig):
        """
        :param initializer: ConfigInitializer used to fetch the dataset and classes.
        :param backbone_cfg: Configuration for the VLM backbone.
        """
        ds = initializer.get_dataset('test', loaders=False, labels=False)
        self.classes = ds.classes
        label_space = {k: [c for c in v['values'].keys()] for k, v in self.classes.items()}
        self.label_map = {k: v['values'] for k, v in self.classes.items()}
        backbone = initializer.initialize(vlm_backbone_cfg)
        preprocessor = DynamicVLMPreprocessor()
        prompt_builder = ClosedLabelTaskBuilder(label_space)
        self.classifier = VLMClassifierJudge(backbone, preprocessor, prompt_builder)

    def __call__(self, X, frame_level=False):
        """
        Classify a batch of sequences using a VLM backend.

        :param X: Input tensor (typically batch of video clips).
        :param frame_level: Not supported for VLMs.
        :return: Dictionary mapping factor names to predicted class index tensors.
        """
        if frame_level:
            raise NotImplementedError("Frame-level classification is not supported for VLMJudge.")
        predictions = self.classifier(X.detach().cpu().numpy())
        labels = {f: torch.tensor([self.label_map[f].get(predictions[i][f], -1) for i in range(len(predictions))]) for f in predictions[0].keys()}
        return labels


    def to(self, device):
        """
        Dummy `.to()` method for compatibility with PyTorch modules.

        :param device: Device to move to (no-op here).
        :return: Self.
        """
        return self