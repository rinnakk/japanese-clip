# Japanese-CLIP
![rinna-icon](./data/rinna.png)

This repository includes codes for Japanese [CLIP (Contrastive Language-Image Pre-Training)](https://arxiv.org/abs/2103.00020) variants by [rinna Co., Ltd](https://rinna.co.jp/).

| Table of Contents |
|-|
| [Pretrained Models](#Pretrained-Models) |
| [Usage](#Usage) |



## Pretrained models

| Model Name | TOP1\* |  TOP5\* |
|:--------:|:--:|:---:|
| [rinna/japanese-cloob-vit-b-16](https://huggingface.co/rinna/japanese-cloob-vit-b-16) | 48.37 | 65.40 | 
| [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16) | 41.09 | 61.83 |
| | | |
| [sonoisa/clip-vit-b-32-japanese-v1](https://huggingface.co/sonoisa/clip-vit-b-32-japanese-v1) | 38.38 | 59.93 |
| [multilingual-CLIP](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) | 14.09 | 26.43 |

*\*Zero-shot ImageNet validation set top-k accuracy. Used `{japanese_class_name}の写真` as text prompts*

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
@misc{japanese-clip,
  author = {rinna Co., Ltd.},
  title = {{Japanese CLIP}},
  howpublished = {\url{https://github.com/rinnakk/japanese-clip}},
  year = 2022,
  month = May
}
```

## License
[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)