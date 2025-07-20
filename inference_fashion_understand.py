from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import os
import json
from tqdm import tqdm
import argparse
import shortuuid
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
from blip3o.conversation import conv_templates, SeparatorStyle
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
from datetime import datetime
import re, random

model_path = sys.argv[1]
diffusion_path = model_path + "/diffusion-decoder"



processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


device_1 = 0


disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)




pipe = DiffusionPipeline.from_pretrained(
   diffusion_path,
   custom_pipeline="pipeline_llava_gen",
   torch_dtype=torch.bfloat16,
   use_safetensors=True,
   variant="bf16",
   multimodal_encoder=multi_model,
   tokenizer=tokenizer,
   safety_checker=None
)


pipe.vae.to(f'cuda:{device_1}')
pipe.unet.to(f'cuda:{device_1}')



def process_image(prompt: str, img: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    text_prompt_for_qwen = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt_for_qwen],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(f'cuda:{device_1}')
    generated_ids = multi_model.generate(**inputs, max_new_tokens=1024)
    input_token_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text



def get_command_args():
    """command line arguments to control the run"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '-i', 
        # help='pattern specification JSON file. File name should end with "_specification.json"', 
        type=str, 
        default='./example_data/evaluation/sketch_sample/man_1.png')
    parser.add_argument(
        '--prompt', '-p', 
        # help='Path to simulation config', 
        type=str, 
        default='./example_data/prompt.txt')
    parser.add_argument(
        '--output', '-o', 
        # help='Path to simulation config', 
        type=str, 
        default='./Logs')

    args = parser.parse_args()
    print('Commandline arguments: ', args)

    return args

def save_output(img: Image.Image, prompt: str, text: str, output_dir: str):
    now_time = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    output_path = os.path.join(output_dir, now_time)
    os.makedirs(output_path, exist_ok=True)

    img.save(os.path.join(output_path, 'img.png'))

    with open(os.path.join(output_path, 'prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(prompt.strip())

    with open(os.path.join(output_path, 'caption.txt'), 'w', encoding='utf-8') as f:
        f.write(text.strip())


if __name__ == "__main__":
    args = get_command_args()
    # === IMAGE UNDERSTANDING DEMO ===
    img = Image.open(args.image).convert("RGB")
    with open(args.prompt, 'r') as f:
        prompt = f.read()
    text = process_image(prompt, img)      
    print(text)
    save_output(img, prompt, text, args.output)

