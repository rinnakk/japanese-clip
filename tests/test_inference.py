import os
from os.path import abspath, dirname, join, realpath
import torch
from PIL import Image
from dotenv import load_dotenv
import japanese_clip as ja_clip


base_dir = dirname(dirname(__file__))
load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


def test_inference_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = ja_clip.load('rinna/japanese-clip-vit-b-16', device=device, use_auth_token=ACCESS_TOKEN)
    tokenizer = ja_clip.load_tokenizer()

    # https://gahag.net/011423-jack-russell-terrier/
    image = preprocess(Image.open(os.path.join(base_dir, "data/dog.jpeg"))).unsqueeze(0).to(device)
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

    assert text_probs[0] == [1.0, 0.0, 0.0]


def test_inference_cloob():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = ja_clip.load('rinna/japanese-cloob-vit-b-16', device=device, use_auth_token=ACCESS_TOKEN)
    tokenizer = ja_clip.load_tokenizer()

    # https://gahag.net/011423-jack-russell-terrier/
    image = preprocess(Image.open(os.path.join(base_dir, "data/dog.jpeg"))).unsqueeze(0).to(device)
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

    assert text_probs[0] == [1.0, 0.0, 0.0]
