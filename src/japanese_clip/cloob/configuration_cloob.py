import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig


logger = logging.get_logger(__name__)


class CLOOBConfig(PretrainedConfig):
    model_type = "cloob"
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        vision_model_type = vision_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        if vision_model_type == "clip":
            self.vision_config = AutoConfig.for_model(
                vision_model_type, **vision_config
            ).vision_config
        elif vision_model_type == "clip_vision_model":
            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(
                vision_model_type, **vision_config
            )

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_configs(
        cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs
    ):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
