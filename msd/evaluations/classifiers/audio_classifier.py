from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from msd.evaluations.classifiers.abstract_classifier import AbstractClassifier


class AudioClassifier(AbstractClassifier):
    """
    A convolutional classifier for audio inputs using Mel-spectrogram features.

    Designed to classify static audio-related features from temporal audio signals.
    """

    def __init__(self, classes: Dict):
        """
        :param classes: Dictionary of all available classes, including metadata.
        """
        super(AudioClassifier, self).__init__()
        self.classes = {k: v for k, v in classes.items() if not v['ignore']}
        self.transform = MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=640, f_max=8000, n_mels=128, power=2.0, norm='slaney')

        # 2D convolutional stack for processing spectrogram input
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Input: [B, 1, 128, T]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Downsample both frequency and time

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 128, 1, 1]
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
        Apply log-mel spectrogram transformation to raw waveform input.

        :param _x: Input waveform tensor of shape [B, T].
        :return: Transformed tensor of shape [B, 1, Mels, Frames].
        """
        x = torch.log1p(self.transform(_x))
        return x

    def forward(self, _x: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Forward pass through feature extractor and classification heads.

        :param _x: Input tensor of shape [B, T].
        :return: Tuple of (sequence-level predictions, empty dict).
        """
        x = self.feature_extractor(_x.unsqueeze(1))  # Shape: [B, 1, Mels, Time]
        x = x.view(x.size(0), -1)  # Flatten to [B, Hidden Dim]

        prediction_logits = {
            feature_name: head(x) for feature_name, head in self.heads.items()
        }
        return prediction_logits, {}
