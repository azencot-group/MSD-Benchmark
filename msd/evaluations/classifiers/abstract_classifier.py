from typing import Dict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from msd.configurations.msd_component import MSDComponent


class AbstractClassifier(ABC, nn.Module, MSDComponent):
    """
    Abstract base class for classifiers used in disentanglement evaluation.

    Each classifier should implement `forward()` to return predictions for:
    - sequence-level (aggregated) classification
    - frame-level (per-frame) classification

    The `classify()` method handles preprocessing, forwarding, and argmax decoding.
    """

    def preprocess(self, _x: torch.Tensor) -> torch.Tensor:
        """
        Optional preprocessing hook applied before the forward pass.

        :param _x: Input tensor of shape [B, T, ...] or [B, ...].
        :return: Preprocessed input tensor.
        """
        return _x

    def classify(self, x: torch.Tensor, frame_level: bool = False) -> Dict[str, torch.Tensor]:
        """
        Run the model and return argmax predictions.

        :param x: Input tensor.
        :param frame_level: Whether to return frame-level (True) or sequence-level (False) predictions.
        :return: Dictionary mapping feature names to class indices.
        """
        x = self.preprocess(x)
        sequence_pred, static_pred = self.forward(x)
        out = static_pred if frame_level else sequence_pred
        return {feature_name: torch.argmax(p, dim=-1) for feature_name, p in out.items()}


    @abstractmethod
    def forward(self, x: torch.Tensor) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """
        Forward pass of the classifier.

        :param x: Input tensor.
        :return: Tuple of (sequence-level predictions, frame-level predictions),
                 each a dict of {feature_name: logits}.
        """
        pass