from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from msd.evaluations.classifiers.abstract_classifier import AbstractClassifier


class AudioClassifier(AbstractClassifier):
    """
    A convolutional classifier for audio inputs using Mel-spectrogram features.

    Designed to classify static audio-related features from temporal audio signals.
    """

    def __init__(self,
                 classes: Dict,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 f_max: int = 8000,
                 power: float = 2.0,
                 norm: str = 'slaney'):
        """
        :param classes: Dictionary of all available classes, including metadata.
        """
        super(AudioClassifier, self).__init__()
        self.classes = {k: v for k, v in classes.items() if not v['ignore']}
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=sample_rate,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           n_mels=n_mels,
                           f_max=f_max,
                           power=power,
                           norm=norm),
            AmplitudeToDB()
        )

        # 2D convolutional stack with temporal pooling preserved
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((None, 1))  # Preserve time, pool frequency
        )

        self.temporal_model = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        def create_heads(classes_):
            return nn.ModuleDict({
                feature_name: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, v['n_classes']),
                    nn.Softmax(dim=1)
                )
                for feature_name, v in classes_.items()
            })

        self.heads = create_heads(self.classes)

    def preprocess(self, _x: torch.Tensor) -> torch.Tensor:
        """
        Apply log-mel spectrogram transformation to raw waveform input.

        :param _x: Input waveform tensor of shape [B, T].
        :return: Transformed tensor of shape [B, 1, Mels, Frames].
        """
        x = self.transform(_x)
        return x

    def forward(self, _x: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Forward pass through feature extractor and classification heads.

        :param _x: Input tensor of shape [B, T].
        :return: Tuple of (sequence-level predictions, empty dict).
        """
        x = self.feature_extractor(_x.unsqueeze(1))  # [B, 1, Mels, Time] → [B, 256, T, 1]
        x = x.squeeze(-1).transpose(1, 2)  # [B, 256, T] → [B, T, 256]

        rnn_out, _ = self.temporal_model(x)  # [B, T, 256]
        x_pooled = rnn_out.mean(dim=1)  # [B, 256]

        predictions = {
            feature_name: head(x_pooled) for feature_name, head in self.heads.items()
        }

        return predictions, {}
