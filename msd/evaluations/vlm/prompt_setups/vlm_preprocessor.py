from abc import ABC, abstractmethod

import base64
import cv2
from io import BytesIO

import numpy as np
from PIL import Image

from msd.configurations.msd_component import MSDComponent


def image2base64(image_array: np.ndarray) -> str:
    """
    Convert a NumPy image array to a base64-encoded PNG string.

    :param image_array: Image as a NumPy array in BGR format.
    :return: Base64-encoded PNG image string.
    """
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class VLMPreprocessor(ABC, MSDComponent):
    """
    Abstract base class for image/video preprocessing for VLM input.

    This class handles image normalization, channel reordering, and encoding into base64.
    Subclasses define how the input is reduced (e.g., selecting a single frame or stacking multiple frames).
    """

    def __init__(self, encoding_fn=image2base64):
        """
        :param encoding_fn: A function that converts a processed image to a base64 string.
        """
        self.encoding_fn = encoding_fn

    def __call__(self, X: np.ndarray) -> str:
        """
        Allows the instance to be called like a function.

        :param X: Input tensor with shape [B, C, H, W].
        :return: Base64-encoded string of the processed image or video.
        """
        return self.process(X)


    @abstractmethod
    def _process(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce or transform the input image(s) prior to encoding.

        :param x: A 4D image tensor of shape [B, H, W, C] in uint8 format.
        :return: A 3D image array (H, W, C) to be encoded.
        """
        pass

    def process(self, x: np.ndarray) -> str:
        """
        Apply preprocessing and encode the result to base64.

        :param x: Tensor of shape [B, C, H, W] with normalized pixel values in [0, 1].
        :return: Base64-encoded image string.
        """
        x = x[:, [2, 1, 0], :, :]  # Convert RGB to BGR
        x = (x * 255).transpose((0, 2, 3, 1)).astype(np.uint8)
        x_enc = self.encoding_fn(self._process(x))
        return x_enc

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        A system prompt to guide the VLM about the type of image input.

        :return: A descriptive prompt string.
        """
        pass

class StaticVLMPreprocessor(VLMPreprocessor):
    """
    Preprocessor for static image input.

    Uses only the first frame of the input batch.
    """

    def _process(self, x: np.ndarray) -> np.ndarray:
        """
        Select the first image from the batch.

        :param x: A batch of images [B, H, W, C].
        :return: A single image [H, W, C].
        """
        return x[0]

    @property
    def system_prompt(self) -> str:
        """
        System prompt describing that the model processes single images.

        :return: Instructional string for static image analysis.
        """
        return "You analyze images and return structured descriptions as JSON."


class DynamicVLMPreprocessor(VLMPreprocessor):
    """
    Preprocessor for dynamic input (videos or sequences of frames).

    Concatenates all frames horizontally to simulate a temporal sequence.
    """

    def _process(self, x: np.ndarray) -> np.ndarray:
        """
        Concatenate all frames side by side.

        :param x: A batch of images [B, H, W, C].
        :return: A single wide image [H, B*W, C].
        """
        return np.concatenate(x, axis=1)

    @property
    def system_prompt(self) -> str:
        """
        System prompt describing that the model processes frame sequences.

        :return: Instructional string for dynamic visual input.
        """
        return (
            "You analyze video frame sequences. "
            "Frames are stacked left to right in order. "
            "Your goal is to return structured descriptions as JSON."
        )