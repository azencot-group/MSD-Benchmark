import base64
import json
from io import BytesIO
from typing import Dict, Iterable, List, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from msd.evaluations.vlm.backbone.vlm_backbone import VLMBackbone


class QwenVLBackbone(VLMBackbone):
    def __init__(self, model_name='Qwen/Qwen2.5-VL-7B-Instruct', max_tokens=128, device=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto").eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _decode_image(self, image64: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(image64)))

    def generate_response(self, prompt: Iterable) -> Dict:
        try:
            text = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return self.extract_json(output_text)

        except Exception as e:
            return {"error": f"Error occurred: {e}"}

    def generate_system_prompt(self, text: str) -> Dict:
        return {"role": "system", "content": text}

    def generate_user_prompt(self, text: str, image64: Union[str, List[str]] = None) -> Dict:
        if image64 is not None:
            if isinstance(image64, list):
                content = [{"type": "image", "image": f"data:image;base64,{im}"} for im in image64]
            else:
                content = [{"type": "image", "image": f"data:image;base64,{image64}"}]
            content.append({"type": "text", "text": text})
            return {"role": "user", "content": content}
        else:
            return {"role": "user", "content": text}

    def generate_examples_prompt(self, X, Y, text: str) -> List[Dict]:
        messages = []
        for image64, labels in zip(X, Y):
            messages.append(self.generate_user_prompt(text, image64))
            messages.append({"role": "assistant", "content": json.dumps(labels)})
        return messages
