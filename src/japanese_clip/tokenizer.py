from typing import Union, List, Dict
import torch
from transformers import T5Tokenizer


def load_tokenizer():
    """
    https://huggingface.co/rinna/japanese-roberta-base
    """
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    return tokenizer


def tokenize(
        texts: Union[str, List[str]],
        tokenizer: T5Tokenizer = None,
        max_seq_len: int = 77,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    This is a function that have the original clip's code has.
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L195
    """
    if isinstance(texts, str):
        texts = [texts]
    if tokenizer is None:
        tokenizer = load_tokenizer()
    inputs = tokenizer(
        texts,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add cls token at first place
    input_ids = [[tokenizer.cls_token_id] + ids for ids in inputs['input_ids']]
    attention_mask = [[1] + am for am in inputs['attention_mask']]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    position_ids = torch.tensor(position_ids, dtype=torch.long)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
    }
