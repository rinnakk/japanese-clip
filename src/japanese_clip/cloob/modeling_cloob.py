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

from typing import Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.clip.modeling_clip import (
    CLIPVisionConfig,
    CLIPVisionModel,
)
from .configuration_cloob import CLOOBConfig
from .loss import cloob_loss


logger = logging.get_logger(__name__)


@dataclass
class CLOOBOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    inv_tau: Union[torch.FloatTensor, float] = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLOOBModel(PreTrainedModel):
    config_class = CLOOBConfig
    base_model_prefix = "cloob"

    def __init__(
        self,
        config: Optional[CLOOBConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        init_inv_tau: float = 14.3,
        learnable_inv_tau: bool = True,
    ):

        if config is None and (vision_model is None or text_model is None):
            raise ValueError(
                "Either a configuration or an vision and a text model has to be provided"
            )

        if config is None:
            config = CLOOBConfig.from_vision_text_configs(
                vision_model.config, text_model.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"config: {config} has to be of type {self.config_class}"
                )

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config, add_pooling_layer=False)
            else:
                vision_model = AutoModel.from_config(config.vision_config, add_pooling_layer=False)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config, add_pooling_layer=False)

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )

        # Logit scales for the inner product in the InfoNCE loss
        # self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
        # self.logit_inv_tau.requires_grad = learnable_inv_tau

        # inv_tau and scale_hopfield are constant as crowsonkb did
        # https://github.com/crowsonkb/cloob-training/blob/master/cloob_training/pretrained_configs/cloob_laion_400m_vit_b_16_16_epochs.json#L5
        self.inv_tau = 30.0
        self.scale_hopfield = 15.0

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        out=False
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(pooled_output)
        if out:
            return text_features, text_outputs
        return text_features

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        ######################
        # pooled_output = vision_outputs[1]  # pooler_output
        pooled_output = vision_outputs.last_hidden_state[:, 0, :]
        ######################
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        loss = None
        if return_loss:
            loss = cloob_loss(image_embeds, text_embeds, self.inv_tau, self.scale_hopfield)

        if not return_dict:
            output = (
                text_embeds,
                image_embeds,
                self.inv_tau,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLOOBOutput(
            loss=loss,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            inv_tau=self.inv_tau, #self.logit_inv_tau.exp(),
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_vision = {
            argument[len("vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(
                    vision_model_name_or_path, add_pooling_layer=False, *model_args, **kwargs_vision
                )
                # TODO: Should we use the pre-trained projection as well ?
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(
                    vision_model_name_or_path, add_pooling_layer=False, *model_args, **kwargs_vision
                )

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(
                text_model_name_or_path, add_pooling_layer=False, *model_args, **kwargs_text
            )

        # instantiate config with corresponding kwargs
        config = CLOOBConfig.from_vision_text_configs(
            vision_model.config, text_model.config, **kwargs
        )

        # init model
        model = cls(config=config, vision_model=vision_model, text_model=text_model)

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        logger.warning(
            "The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight', 'logit_scale']` "
            "are newly initialized. You should probably TRAIN this model on a down-stream task "
            "to be able to use it for predictions and inference."
        )

        return model
