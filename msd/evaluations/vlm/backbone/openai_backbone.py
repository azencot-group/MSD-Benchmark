import json
from pathlib import Path
from typing import Iterable, Dict, List, Union, Optional

from openai import OpenAI

from msd.evaluations.vlm.backbone.vlm_backbone import VLMBackbone, extract_json


class OpenAIGPTBackbone(VLMBackbone):
    """
    OpenAI-based implementation of a VLM backbone using GPT-4 or similar models.

    Supports text-only and image-text prompting via base64 strings.
    """

    def __init__(self, model: str = 'gpt-4o', max_tokens: int = 300, api_key: Optional[str] = None):
        """
        :param model: The name of the model to use (e.g., 'gpt-4o').
        :param max_tokens: Maximum number of tokens for the response.
        :param api_key: Your OpenAI API key.
        """
        if api_key is None:
            api_key = Path.home().joinpath('.openai', 'token.txt')
            api_key = api_key.read_text().strip()
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: Iterable):
        """
        Sends the prompt to the OpenAI API and parses the JSON response.

        :param prompt: A list of role-based messages following OpenAI's chat format.
        :return: Structured response dictionary, or error message if parsing fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_tokens=self.max_tokens
            )
            output_text = response.choices[0].message.content
            # structured_output = json.loads(output_text)
            structured_output = extract_json(output_text)

            return structured_output
        except Exception as e:
            return {"error": f"Error occurred: {e}"}

    def generate_system_prompt(self, text: str) -> Dict:
        """
        Creates a system message prompt for OpenAI chat.

        :param text: Content of the system instruction.
        :return: Formatted dictionary for system role.
        """
        return {"role": "system", "content": text}

    def generate_user_prompt(self, text: str, image64: Union[str, List[str]] = None) -> Dict:
        """
        Creates a user message prompt, with optional image(s).

        :param text: Instruction or query text.
        :param image64: Base64 string or list of base64 strings for image input.
        :return: Formatted user message with text and optional images.
        """
        if image64 is not None:
            if isinstance(image64, list):
                content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{im}"}} for im in image64]
            else:
                content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image64}"}}]
            content.append({"type": "text", "text": text})
            return {"role": "user", "content": content}
        else:
            return {"role": "user", "content": text}

    def generate_examples_prompt(self, X, Y, text: str) -> List[Dict]:
        """
        Builds a list of prompt-response examples from given images and labels.

        :param X: List of base64-encoded image strings.
        :param Y: List of corresponding label dictionaries.
        :param text: Prompt text to accompany each image.
        :return: List of alternating user and assistant messages.
        """
        messages = []
        for image64, labels in zip(X, Y):
            messages.append(self.generate_user_prompt(text, image64))
            messages.append({"role": "assistant", "content": json.dumps(labels)})
        return messages
