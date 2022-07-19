# Japanese-CLIP
![rinna-icon](./data/rinna.png)

This repository includes codes for Japanese [CLIP (Contrastive Language-Image Pre-Training)](https://arxiv.org/abs/2103.00020) variants by [rinna Co., Ltd](https://rinna.co.jp/).

| Table of Contents |
|-|
| [News](#news) |
| [Pretrained Models](#Pretrained-Models) |
| [Usage](#Usage) |
| [Citation](#Citation) |
| [License](#License) |

## News
### July 2022
v0.2.0 was released!
- Both CLIP and CLOOB models were upgraded! Now, `rinna/japanese-cloob-vit-b-16` achieves 54.64.
- Released our Japanese prompt templates and an example code (see `scripts/example.py`) for zero-shot ImageNet classification. Those templates were cleaned for Japanese based on the [OpenAI 80 templates](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
- Changed the citation


## Pretrained models

| Model Name | TOP1\* |  TOP5\* |
|:--------:|:--:|:---:|
| [rinna/japanese-cloob-vit-b-16](https://huggingface.co/rinna/japanese-cloob-vit-b-16) | 54.64 | 72.86 | 
| [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16) | 50.69 | 72.35 |
| | | |
| [sonoisa/clip-vit-b-32-japanese-v1](https://huggingface.co/sonoisa/clip-vit-b-32-japanese-v1) | 38.88 | 60.71 |
| [multilingual-CLIP](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) | 14.36 | 27.28 |

*\*Zero-shot ImageNet validation set top-k accuracy.*

## Usage

1. Install package
```shell
$ pip install git+https://github.com/rinnakk/japanese-clip.git
```
2. Run
```python
from PIL import Image
import torch
import japanese_clip as ja_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
# ja_clip.available_models()
# ['rinna/japanese-clip-vit-b-16', 'rinna/japanese-cloob-vit-b-16']
# If you want v0.1.0 models, set `revision='v0.1.0'`
model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

image = preprocess(Image.open("./data/dog.jpeg")).unsqueeze(0).to(device)
encodings = ja_clip.tokenize(
    texts=["犬", "猫", "象"],
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**encodings)
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1.0, 0.0, 0.0]]
```

## Citation 
To cite this repository:
```shell
@inproceedings{japanese-clip,
  author = {シーン 誠, 趙 天雨, 沢田 慶},
  title = {日本語における言語画像事前学習モデルの構築と公開},
  booktitle= {The 25th Meeting on Image Recognition and Understanding},
  year = 2022,
  month = July,
}
```

## License
[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)