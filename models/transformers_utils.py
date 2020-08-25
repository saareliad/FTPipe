from .transformers_cfg import MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS, MODEL_TYPES
from .models import CommonModelHandler
from transformers import AutoModel, AutoConfig, AutoTokenizer
from enum import Enum, auto
import os


class GetConfigFrom(Enum):
    HardCoded = auto()
    ParsedArgs = auto()
    Generated = auto()


class HFModelHandler(CommonModelHandler):
    def __init__(self,
                 method: GetConfigFrom = GetConfigFrom.HardCoded,
                 *args, **kw):
        super().__init__(*args, **kw)
        self.pipeline_transformer_config = None
        self.method = method

        self.tokenizer = None
        self.config = None

    def get_normal_model_instance(self, *args, **kw):
        cfg = self.get_pipeline_transformer_config()
        model, tokenizer, config = pretrained_model_config_and_tokenizer(**cfg)

        self.tokenizer = tokenizer
        self.config = config
        return model

    def get_pipeline_transformer_config(self):
        if self.pipeline_transformer_config is None:
            if self.method == GetConfigFrom.Generated:
                raise NotImplementedError()
            elif self.method == GetConfigFrom.ParsedArgs:
                raise NotImplementedError()
            elif self.method == GetConfigFrom.HardCoded:
                assert not os.path.exists(self.generated_file_name_or_path)
                cfg = MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.get(self.generated_file_name_or_path)()
            self.pipeline_transformer_config = cfg
        return self.pipeline_transformer_config

    def get_extra(self, *args, **kw):
        return dict(config=self.config, tokenizer=self.tokenizer)

    def get_loader(self, *args, **kw):
        raise NotImplementedError()


def resize_token_embeddings(model, tokenizer):
    # NOTE: must use the same tokenizer used at partitioning.
    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))


def pretrained_model_config_and_tokenizer(
        model_type: str,
        model_name_or_path: str,
        config_name: str = "",
        tokenizer_name: str = "",
        do_lower_case: bool = False,
        cache_dir: str = "",
        stateless_tied=False,
        explicitly_set_dict={},
        **config_kw
):
    config_class, model_class, tokenizer_class = MODEL_TYPES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None,
        **config_kw
    )

    for k, v in explicitly_set_dict.items():
        setattr(config, k, v)

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None)

    use_cdn = model_name_or_path in {"t5-11b"}
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
        use_cdn=use_cdn)

    resize_token_embeddings(model, tokenizer)

    if stateless_tied:
        model_to_resize = model.module if hasattr(model, 'module') else model

        if hasattr(model_to_resize, "make_stateless_after_loaded_tied_and_resized"):
            model_to_resize.make_stateless_after_loaded_tied_and_resized()
        elif hasattr(model_to_resize, "make_stateless"):
            # because someone changed the name I gave it and made dangerous code look normal...
            model_to_resize.make_stateless()
        else:
            raise ValueError(f"Problem making model stateless. model_type: {model_type}")

    return model, tokenizer, config
