import io
import requests
import pandas as pd
from PIL import Image
import torch
import torchvision

import japanese_clip as ja_clip
from japanese_clip.utils.callbacks import ImagenetClassificationCallback

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model, tokenizer
    model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    tokenizer = ja_clip.load_tokenizer()

    ##################
    # Text Retrieval #
    ##################
    img = Image.open(io.BytesIO(requests.get("data/dog.jpeg").content))
    image = preprocess(img).unsqueeze(0).to(device)
    encodings = ja_clip.tokenize(
        texts=["犬", "猫", "象"],
        max_seq_len=77,
        device=device,
        tokenizer=tokenizer,  # this is optional. if you didn't pass, load tokenizer each time
    )

    with torch.no_grad():
        image_features = model.get_image_features(image)
        text_features = model.get_text_features(**encodings)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().detach().tolist()

    print("Label probs:", text_probs)
    # prints: [[1.0, 0.0, 0.0]]

    ##################################
    # Zero-shot Image Classification #
    ##################################
    from japanese_clip.utils.imagenet_zeroshot_data import imagenet_templates, imagenet_classnames

    imagenet_data_path = "ImageNet path"

    templates_df = pd.DataFrame.from_dict(imagenet_templates)
    classes_df = pd.DataFrame.from_dict(imagenet_classnames)
    imagenet_classes = classes_df["ja"].values.tolist()
    imagenet_templates_lan = templates_df["ja"].values.tolist()
    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates_lan)} templates")

    dataset = torchvision.datasets.ImageNet(imagenet_data_path, split="val", transform=preprocess)
    imagenet_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        collate_fn=None,
    )
    imagenet_callback = ImagenetClassificationCallback(imagenet_classes, imagenet_templates_lan, imagenet_dataloader)
    result_dict = imagenet_callback.zeroshot(model, tokenizer)
    print(result_dict)
    # prints: {"top1": xx, "top5": xx, "top10": xx, "top100": xx}
