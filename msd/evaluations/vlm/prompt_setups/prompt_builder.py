import json
from abc import abstractmethod, ABC
from typing import List, Dict

from msd.configurations.msd_component import MSDComponent


class VLMTaskBuilder(ABC, MSDComponent):
    """
    Abstract base class for constructing user prompts for VLM-based classification tasks.

    Each subclass must implement a `user_prompt` property, which returns a string prompt
    suitable for submission to a vision-language model (VLM) during evaluation.
    """

    @property
    @abstractmethod
    def user_prompt(self) -> str:
        """
        The VLM-ready user prompt tailored to the specific classification task.

        :return: A prompt string to send to the model.
        """
        pass


class ClassificationTaskBuilder(VLMTaskBuilder):
    """
    Base class for classification-type prompt builders.
    """
    pass


class ContrastiveTaskBuilder(VLMTaskBuilder):
    """
    Base class for prompt builders that require image pair comparisons.
    """
    pass


class ClosedLabelTaskBuilder(ClassificationTaskBuilder):
    """
    Prompt builder for classification tasks where the feature space and label set are known.

    Attributes:
        factors (Dict[str, List[str]]): A dictionary mapping each attribute name to its list of allowed values.
    """

    def __init__(self, factors: Dict[str, List[str]]):
        self.factors = factors

    @property
    def user_prompt(self) -> str:
        """
        Prompt that asks the model to classify known attributes using pre-defined label options.

        :return: A formatted prompt string with allowed values.
        """
        return (
            "You are given an image. Your task is to classify its visual attributes.\n\n"
            "For each attribute listed below, choose the correct value **only** from the allowed options provided.\n"
            "Do not invent or guess values that are not listed.\n\n"
            f"Attributes and their allowed values:\n{json.dumps(self.factors, indent=2)}\n\n"
            "Return your answer in JSON format. The output should be a dictionary where each key is an attribute "
            "and each value is one of the allowed options."
        )



class OpenLabelTaskBuilder(ClassificationTaskBuilder):
    """
    Prompt builder for classification tasks with a known feature space but unknown label sets.

    Attributes:
        factors (List[str]): List of attribute names to describe.
    """
    def __init__(self, factors: List[str]):
        self.factors = factors

    @property
    def user_prompt(self) -> str:
        """
        Prompt asking the model to freely describe known attributes.

        :return: Prompt string instructing freeform value generation in JSON format.
        """
        factors = {k: '' for k in self.factors}
        return (f"Describe the following attributes of the image in your own words\n"
                f"{json.dumps(factors)}\n"
                f"Return your answer in JSON format with keys as attributes and values as descriptions.")

class UnknownFeatureSpaceTaskBuilder(ContrastiveTaskBuilder):
    """
    Prompt builder for contrastive classification tasks with an unknown feature space.

    The model compares two images and identifies attributes that are different and the same.
    """

    @property
    def user_prompt(self) -> str:
        """
        Prompt that instructs the model to infer and describe differences and similarities between two images.

        :return: Prompt formatted with instructions and example output schema.
        """
        return (
            "You are given two images. Analyze them carefully and identify:\n"
            "1. A dictionary of attributes that differ between the images. Each key should be a short string that combines the object and attribute "
            "(e.g., 'shirt color' or 'box size'), and the value should describe how the attribute differs between the two images.\n"
            "2. A dictionary of attributes that remain the same in both images, using the same format.\n\n"
            "Return your answer in JSON format with two fields: 'different' and 'same'. Each should be a dictionary where:\n"
            "- the keys are strings like 'hair color', 'pose type', or 'floor hue'\n"
            "- the values are short, clear descriptions of the attribute (or difference)\n\n"
            "Example format:\n"
            "{\n"
            "  \"different\": {\n"
            "    \"shirt color\": \"red in image 1, blue in image 2\",\n"
            "    \"pants length\": \"shorts in image 1, long pants in image 2\"\n"
            "  },\n"
            "  \"same\": {\n"
            "    \"hair style\": \"curly\",\n"
            "    \"pose type\": \"walking\"\n"
            "  }\n"
            "}"
        )

class UnknownLabelSpaceTaskBuilder(ContrastiveTaskBuilder):
    """
    Prompt builder for contrastive classification tasks with a known feature space
    but unknown labels.

    The model is given a predefined list of features and must determine,
    for each feature, whether it is the same or different between two images,
    and provide a short explanation.
    """

    def __init__(self, features: list[str]):
        """
        Initialize with the known list of features.

        :param features: List of feature names (e.g., ["hair color", "object shape", "character pose"]).
        """
        self.features = features

    @property
    def user_prompt(self) -> str:
        """
        Prompt that instructs the model to analyze the two images according to known features
        and classify each feature into 'same' or 'different' with explanations.

        :return: Prompt formatted with instructions and expected output format.
        """
        features_list = "\n".join(f"- {feature}" for feature in self.features)

        return (
            "You are given two images along with a list of predefined features to analyze.\n"
            "For each feature, determine whether it is the same or different between the two images, "
            "and provide a short explanation.\n\n"
            "List of features:\n"
            f"{features_list}\n\n"
            "Return your answer in JSON format with two fields: 'different' and 'same'. Each should be a dictionary where:\n"
            "- the keys are the feature names (exactly as listed above)\n"
            "- the values are short, clear descriptions explaining the similarity or difference\n\n"
            "Example format:\n"
            "{\n"
            "  \"different\": {\n"
            "    \"hair color\": \"black in image 1, blonde in image 2\",\n"
            "    \"shirt color\": \"blue in image 1, red in image 2\"\n"
            "  },\n"
            "  \"same\": {\n"
            "    \"pose\": \"standing\"\n"
            "  }\n"
            "}"
        )

class FactorwiseBinaryComparisonTaskBuilder(ContrastiveTaskBuilder):
    """
    Prompt builder for binary contrastive tasks.

    The model determines, for each known factor, whether it is the same or different between two images.

    Attributes:
        factors (List[str]): The list of known factors to evaluate.
    """

    def __init__(self, factors: list[str]):
        self.factors = factors

    @property
    def user_prompt(self):
        """
        Prompt asking the model to compare each factor between two images and return 1 (same) or 0 (different).

        :return: Formatted prompt string listing the factors.
        """
        formatted_factors = "\n".join(f"- {factor}" for factor in self.factors)
        return (
            "You are given two images. Your task is to examine the images and evaluate each of the following factors:\n"
            f"{formatted_factors}\n\n"
            "For each factor, output 1 if the factor is the same in both images, or 0 if it is different.\n"
            "Return your answer as a JSON object where each key is a factor name, and the value is either 0 or 1."
        )