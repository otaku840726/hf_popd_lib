import torch
from transformers import Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import io
import base64
import requests
import json
import re
from datetime import datetime
from numpy import uint8, exp

ocr_models = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
]

def get_ocr_model(model_id):
    if model_id == "Qwen/Qwen2-VL-7B-Instruct":
        return Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, torch_dtype="auto").cuda().eval()
    elif model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        return MllamaForConditionalGeneration.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

def get_ocr_processors(model_id):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_ocr_cf(image, text_input=None, temperature=None, min_p=None, model_id=None):
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"
    ACCOUNT_ID = os.getenv("CF_ID")
    AUTH_TOKEN = os.getenv("CF_TOKEN")

    prompt = f"{user_prompt}<|image_1|>\n{text_input}{prompt_suffix}{assistant_prompt}"
    image_buffer = io.BytesIO()
    img = Image.fromarray(uint8(image))
    img.save(image_buffer, format='PNG')
    image_ba = bytearray(image_buffer.getvalue())

    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a friendly assistant"},
                {"role": "user", "content": """
                Please help me check if this is a receipt for a bank transaction.
                I need the following data,
                ...
                """}
            ],
            "image": list(image_ba)
        }
    )
    result = response.json()
    json_regex = r'\{.*\}'
    match = re.search(json_regex, result['result']['response'], re.DOTALL)
    if match:
        json_string = match.group(0)
        data = json.loads(json_string)
        result = data
    else:
        result = json.dumps({}, indent=4)
    return result
