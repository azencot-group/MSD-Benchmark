from typing import Dict, Tuple

import torch
import torch.nn as nn

from msd.evaluations.classifiers.abstract_classifier import AbstractClassifier


class TimeSeriesClassifier(AbstractClassifier):
    """
    A 1D convolutional classifier for multivariate time-series data.

    Designed for classification of static properties based on temporal sensor sequences.
    """

    def __init__(self, classes: Dict[str, Dict[str, any]]):
        """
        :param classes: Dictionary of feature names with metadata and class count.
        """
        super(TimeSeriesClassifier, self).__init__()
        self.classes = {k: v for k, v in classes.items() if not v['ignore']}
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1),  # [B, C, T]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample T dimension

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output: [B, 256, 1]
        )

        def create_heads(classes_):
            return nn.ModuleDict({feature_name: nn.Sequential(
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128, v['n_classes']),
                nn.Softmax(dim=1)
            ) for feature_name, v in classes_.items()})

        self.heads = create_heads(self.classes)

    def preprocess(self, _x: torch.Tensor) -> torch.Tensor:
        """
        Permute input tensor for Conv1D. Expects input of shape [B, T, C] and returns [B, C, T].

        :param _x: Input time-series batch.
        :return: Transformed tensor ready for 1D convolutions.
        """
        return _x.permute(0, 2, 1)  # [B, C, T]

    def forward(self, _x: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Forward pass through the time-series classifier.

        :param _x: Input tensor of shape [B, C, T].
        :return: Tuple of (sequence-level predictions, empty dict).
        """
        x = self.feature_extractor(_x)  # [B, 256, 1]
        x = x.squeeze(-1)  # [B, 256] Flatten temporal dimension

        prediction_logits = {
            feature_name: head(x) for feature_name, head in self.heads.items()
        }
        return prediction_logits, {}