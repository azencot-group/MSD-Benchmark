import torch
import tensorflow as tf
import tensorflow_hub as hub

class MelSpecEncoder:
    """
    Utility class for encoding and decoding audio using a Mel-spectrogram representation.

    This encoder is specifically configured for compatibility with the SoundStream
    mel decoder model from TensorFlow Hub.
    """

    def __init__(self):
        """
        Initialize Mel-spectrogram parameters and load the inverse mel decoder model.
        """
        self.SAMPLE_RATE = 16000
        self.N_FFT = 1024
        self.HOP_LENGTH = 320
        self.WIN_LENGTH = 640
        self.N_MEL_CHANNELS = 128
        self.MEL_FMIN = 0.0
        self.MEL_FMAX = int(self.SAMPLE_RATE // 2)
        self.CLIP_VALUE_MIN = 1e-5
        self.CLIP_VALUE_MAX = 1e8

        # Mel filterbank matrix
        self.MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.N_MEL_CHANNELS,
            num_spectrogram_bins=self.N_FFT // 2 + 1,
            sample_rate=self.SAMPLE_RATE,
            lower_edge_hertz=self.MEL_FMIN,
            upper_edge_hertz=self.MEL_FMAX)

        # Pretrained inverse mel decoder (SoundStream)
        self.inv_mel = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/frameworks/TensorFlow2/variations/mel-decoder-music/versions/1')

    def calc_melspec(self, x: tf.Tensor) -> tf.Tensor:
        """
        Calculate log-Mel spectrogram from waveform.

        :param x: Tensor of shape [B, T] with float32 values in range [-1, 1].
        :return: Tensor of shape [B, Time, Mels] normalized to [0, 1].
        """
        fft = tf.signal.stft(
            x,
            frame_length=self.WIN_LENGTH,
            frame_step=self.HOP_LENGTH,
            fft_length=self.N_FFT,
            window_fn=tf.signal.hann_window,
            pad_end=True)
        fft_modulus = tf.abs(fft)

        output = tf.matmul(fft_modulus, self.MEL_BASIS)

        output = tf.clip_by_value(
            output,
            clip_value_min=self.CLIP_VALUE_MIN,
            clip_value_max=self.CLIP_VALUE_MAX)
        output = tf.math.log(output)
        output = output / tf.math.log(self.CLIP_VALUE_MAX)
        return output

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw audio waveform (PyTorch) into a log-Mel spectrogram (PyTorch).

        :param x: Input waveform tensor of shape [B, T] on any device.
        :return: Log-mel spectrogram tensor of shape [B, Mels, Time] on the same device.
        """
        device = x.device
        arr_tensor = tf.convert_to_tensor(x.detach().cpu().numpy())
        spectrogram = self.calc_melspec(arr_tensor).numpy()
        return torch.from_numpy(spectrogram).permute(0, 2, 1).to(device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from log-Mel spectrogram using the inverse model.

        :param x: Log-mel spectrogram tensor of shape [B, Mels, Time] on any device.
        :return: Reconstructed waveform tensor of shape [B, T] on the same device.
        """
        device = x.device
        spectrogram = tf.convert_to_tensor(x.detach().cpu().numpy().transpose(0, 2, 1))
        spectrogram = spectrogram * tf.math.log(self.CLIP_VALUE_MAX)
        with tf.device('/cpu:0'):
            wav = self.inv_mel(spectrogram).numpy()
        wav = torch.from_numpy(wav).to(device)
        return wav