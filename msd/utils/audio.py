import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn
from vocos import Vocos


class MelSpecEncoder(nn.Module):
    """
    PyTorch equivalent of the MelSpecEncoder using torchaudio and vocos.
    """

    def __init__(self, orig_sr: int = 16000):
        """
        :param orig_sr: Original sample rate of the input audio.
        """
        super().__init__()
        self.orig_sr = orig_sr
        self.target_sr = 24000 # Vocos model sample rate
        self.max_value = 1e8
        self.resampler = torchaudio.transforms.Resample(orig_freq=self.orig_sr, new_freq=self.target_sr)
        self.mel_transform = T.MelSpectrogram(
        sample_rate=self.target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        win_length=1024,
        f_min=40.0,
        f_max=12000.0,
        mel_scale="htk",  # matches original config
        power=1.0,  # Vocos expects magnitude
        norm=None)

        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio waveform into a log-Mel spectrogram.

        :param x: Tensor of shape [B, T] with float32 values in range [-1, 1].
        :return: Log-mel spectrogram tensor of shape [B, Mels, Time], normalized to [0, 1].
        """
        x_resampled = self.resampler(x)
        mel = self.mel_transform(x_resampled)
        mel = torch.clamp(mel, min=1e-5)
        mel = torch.log(mel) / np.log(self.max_value)
        return mel

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from log-Mel spectrogram using the inverse model.

        :param x: Log-mel spectrogram tensor of shape [B, Mels, Time], normalized to [0, 1].
        :return: Reconstructed waveform tensor of shape [B, T].
        """
        x = x * np.log(self.max_value)
        y_hat = self.vocos.decode(x)
        denoised = F.gain(y_hat, gain_db=-1.0)
        return denoised