import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Iterable

from msd.configurations.msd_component import MSDComponent


class VLMBackbone(ABC, MSDComponent):
    """
    Abstract base class for Vision-Language Model (VLM) backbones.

    A VLM backbone handles the interaction with a specific multimodal model API
    (e.g., OpenAI, Claude, Gemini), providing utilities to generate prompts and parse responses.
    """

    @abstractmethod
    def generate_response(self, prompt: Iterable) -> Dict:
        """
        Generate a model response from the given prompt.

        :param prompt: A list of formatted messages.
        :return: The raw or structured output from the model.
        """
        pass

    @abstractmethod
    def generate_system_prompt(self, text: str) -> Dict:
        """
        Generate a system prompt for the model.

        :param text: The system instruction text.
        :return: A dictionary formatted for the model.
        """
        pass

    @abstractmethod
    def generate_user_prompt(self, text: str, image64: Union[str,  List[str]] = None) -> Dict:
        """
        Generate a user prompt for the model.

        :param text: The user instruction or query text.
        :param image64: Base64-encoded string or list of strings for image input.
        :return: A dictionary formatted for the model.
        """
        pass

    @abstractmethod
    def generate_examples_prompt(self, X, Y, text: str) -> List[Dict]:
        """
        Generate a list of prompt-response examples for the model.

        :param X: List of base64-encoded image strings.
        :param Y: List of corresponding label dictionaries.
        :param text: The user instruction or query text.
        :return: A list of dictionaries formatted for the model.
        """
        pass

    @staticmethod
    def extract_json(text: str) -> Dict:
        """
        Extracts a JSON object from a string. Strips markdown formatting if present.

        :param text: The string containing JSON content, possibly inside a markdown block.
        :return: The extracted JSON object as a Python dictionary.
        :raises ValueError: If no valid JSON object is found.
        """
        # Remove markdown blocks (```json ... ```)
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try to find a JSON object inside the string
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in text.")