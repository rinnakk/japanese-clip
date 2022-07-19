# coding=utf-8
# Copyright 2022 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
import json
import torch
from torchvision import transforms as T
from huggingface_hub import hf_hub_url, cached_download
import os

from .clip import CLIPModel
from .cloob import CLOOBModel

# TODO: Fill in repo_ids
MODELS = {
    'rinna/japanese-clip-vit-b-16': {
        'repo_id': 'rinna/japanese-clip-vit-b-16',
        'model_class': CLIPModel,
    },
    'rinna/japanese-cloob-vit-b-16': {
        'repo_id': 'rinna/japanese-cloob-vit-b-16',
        'model_class': CLOOBModel,
    }
}
MODEL_CLASSES = {
    "cloob": CLOOBModel,
    "clip": CLIPModel,
}
MODEL_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"


def available_models():
    return list(MODELS.keys())


def _convert_to_rgb(image):
    return image.convert('RGB')


def _transform(image_size):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        _convert_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])


def _download(repo_id: str, cache_dir: str):
    config_file_url = hf_hub_url(repo_id=repo_id, filename=CONFIG_FILE)
    cached_download(config_file_url, cache_dir=cache_dir)
    model_file_url = hf_hub_url(repo_id=repo_id, filename=MODEL_FILE)
    cached_download(model_file_url, cache_dir=cache_dir)


def load(
        model_name: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
):
    """
    Args:
        model_name: model unique name or path to pre-downloaded model
        device: device to put the loaded model
        kwargs: kwargs for huggingface pretrained model class
    Return:
        (torch.nn.Module, A torchvision transform)
    """
    if model_name in MODELS.keys():
        ModelClass = CLIPModel if 'clip' in model_name else CLOOBModel
    elif os.path.exists(model_name):
        assert os.path.exists(os.path.join(model_name, CONFIG_FILE))
        with open(os.path.join(model_name, CONFIG_FILE), "r", encoding="utf-8") as f:
            j = json.load(f)
        ModelClass = MODEL_CLASSES[j["model_type"]]
    else:
        RuntimeError(f"Model {model_name} not found; available models = {available_models()}")

    model = ModelClass.from_pretrained(model_name, **kwargs)
    model = model.eval().requires_grad_(False).to(device)
    return model, _transform(model.config.vision_config.image_size)
