# Japanese-CLIP
![rinna-icon](./data/rinna.png)

This repository contains [CLIP](https://arxiv.org/abs/2103.00020) for Japanese.


| Table of Contents |
|-|
| [Available Models](#Available-Models) |
| [Usage](#Usage) |



## Available Models

Zero-shot ImageNet validation set accuracy:

| Accuracy | `rinna/japanese-cloob-vit-b-16` | `rinna/japanese-clip-vit-b-16` | [sonoisa/clip-vit-b-32-japanese-v1](https://huggingface.co/sonoisa/clip-vit-b-32-japanese-v1) | [multilingual-CLIP](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) |
|:--------:|:--:|:---:|:---:|:---:|
| Accuracy@1 | 48.37 | 41.09 | 38.38 | 14.09 |
| Accuracy@5 | 65.40 | 61.83 | 59.93 | 26.43 |

*Used `{japanese_class_name}の写真` for text prompts* 

## Usage

1. Install package
```shell
$ pip install git+https://github.com/rinnakk/japanese-clip.git
```
2. Run
```python
import torch
import japanese_clip as ja_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# ja_clip.available_models()
# ['rinna/japanese-clip-vit-b-16', 'rinna/japanese-cloob-vit-b-16']
model, preprocess = ja_clip.load("clip_vit_b_16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

image = preprocess(Image.open("./data/dog.jpeg")).unsqueeze(0).to(device)
encodings = ja_clip.tokenize(
    texts=["犬", "猫", "象"],
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you didn't pass, load tokenizer each time
)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**encodings)
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1.0, 0.0, 0.0]]
```
