import os
from enum import Enum, auto

from pipe.models.registery.model_handler import CommonModelHandler, register_model
from pipe.models.transformers_cfg import MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS
from pipe.models.transformers_utils import pretrained_model_config_and_tokenizer


class GetConfigFrom(Enum):
    HardCoded = auto()
    ParsedArgs = auto()
    Generated = auto()


class HFModelHandler(CommonModelHandler):
    def __init__(self,
                 method: GetConfigFrom = GetConfigFrom.HardCoded,
                 *args,
                 **kw):
        super().__init__(*args, **kw)
        self.pipeline_transformer_config = None
        self.method = method

        self.tokenizer = None
        self.config = None

    def _get_normal_model_instance(self, *args, **kw):
        if self.normal_model_instance is None:
            cfg = self.get_pipeline_transformer_config()
            model, tokenizer, config = pretrained_model_config_and_tokenizer(**cfg)

            self.tokenizer = tokenizer
            self.config = config
            self.normal_model_instance = model

        assert hasattr(self, "tokenizer")
        assert hasattr(self, "config")

        return self.normal_model_instance

    def get_pipeline_transformer_config(self):
        if self.pipeline_transformer_config is None:
            if self.method == GetConfigFrom.Generated:
                raise NotImplementedError()
            elif self.method == GetConfigFrom.ParsedArgs:
                raise NotImplementedError()
            elif self.method == GetConfigFrom.HardCoded:
                assert not os.path.exists(self.generated_file_name_or_path)
                cfg = MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.get(
                    self.generated_file_name_or_path)()
            else:
                raise NotImplementedError()
            self.pipeline_transformer_config = cfg
        return self.pipeline_transformer_config

    def get_extra(self, *args, **kw):
        return dict(config=self.config, tokenizer=self.tokenizer)

    def get_loader(self, *args, **kw):
        raise NotImplementedError()


for name in MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS:
    register_model(name,
                   handler=HFModelHandler(generated_file_name_or_path=name,
                                          method=GetConfigFrom.HardCoded))
