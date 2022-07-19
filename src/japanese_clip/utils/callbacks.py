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

from tqdm.auto import tqdm
import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    output = torch.from_numpy(np.asarray(output))
    target = torch.from_numpy(np.asarray(target))
    pred = output.topk(max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


class ImagenetClassificationCallback:
    def __init__(
            self,
            imagenet_classes,
            imagenet_templates,
            imagenet_dataloader,
    ):
        self.imagenet_classes = imagenet_classes
        self.imagenet_templates = imagenet_templates
        self.imagenet_dataloader = imagenet_dataloader

    def tokenize(self, tokenizer, examples, device):
        encoding_inputs = tokenizer(examples, max_length=76, padding="max_length", truncation=True, add_special_tokens=False)
        # add cls token at first place
        input_ids = [[tokenizer.cls_token_id] + ids for ids in encoding_inputs['input_ids']]
        attention_mask = [[1] + am for am in encoding_inputs['attention_mask']]
        position_ids = [list(range(0, len(input_ids[0])))] * len(examples)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def zeroshot_classifier(self, model, tokenizer, classnames, templates):
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            class_embeddings = model.get_text_features(**self.tokenize(tokenizer, texts, model.device)).detach().cpu().numpy()
            class_embeddings = class_embeddings / np.linalg.norm(
                class_embeddings, axis=-1, keepdims=True
            )
            class_embedding = np.mean(class_embeddings, axis=0)
            class_embedding /= np.linalg.norm(class_embedding, axis=-1)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = np.stack(zeroshot_weights, axis=1)
        return zeroshot_weights

    def zeroshot(self, model, tokenizer) -> dict:
        print("Imagenet Zeroshot Classification...")
        zeroshot_weights = self.zeroshot_classifier(model, tokenizer, self.imagenet_classes, self.imagenet_templates)
        top_ns = [1, 5, 10, 100]
        acc_counters = [0.0 for _ in top_ns]
        n = 0.0

        for i, (images, target) in enumerate(tqdm(self.imagenet_dataloader)):
            target = target.numpy()
            # predict
            image_features = model.get_image_features(images.to(model.device)).detach().cpu().numpy()
            image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
            logits = 100.0 * image_features @ zeroshot_weights

            # measure accuracy
            accs = accuracy(logits, target, topk=top_ns)
            for j in range(len(top_ns)):
                acc_counters[j] += accs[j]
            n += images.shape[0]

        tops = {f"imagenet/top{top_ns[i]}": acc_counters[i] / n * 100 for i in range(len(top_ns))}

        return tops

