#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
# from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.lvm.vision_llama import LongVisionLlamaForCausalLM


def load_pretrained_model(model_path, model_base, model_name, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    kwargs['torch_dtype'] = torch.float16

    model = LongVisionLlamaForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def image_processor(images, **kwargs):
        out_images = []
        for image in images:
            image = image.convert('RGB')
            W, H = image.size
            image_size = 256
            ar = W / H
            if W > H:
                new_H = image_size
                new_W = int(image_size * ar)
            else:
                new_W = image_size
                new_H = int(image_size / ar)
            image = image.resize((new_W, new_H))
            left = (new_W - image_size) // 2
            top = (new_H - image_size) // 2
            right = (new_W + image_size) // 2
            bottom = (new_H + image_size) // 2
            image = image.crop((left, top, right, bottom))
            image = torch.FloatTensor(image).movedim(-1, 0) / 127.5 - 1
            out_images.append(image)
        out_images = torch.stack(out_images, dim=0)
        return out_images

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
