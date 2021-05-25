from enum import Enum, auto

from .transformers_cfg import MODEL_TYPES
import transformers

class GetConfigFrom(Enum):
    HardCoded = auto()
    ParsedArgs = auto()
    Generated = auto()


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
        do_resize_token_embedding=True,
        explicitly_set_dict={},
        **config_kw
):
    # NOTE its not AutoModel because we sometimes slightly modify the model.
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

    # use_cdn = model_name_or_path not in {"t5-11b"}
    extra_kwargs = {}
    if model_name_or_path in {"t5-11b"} and transformers.__version__ < ('4.1.1'):
        extra_kwargs['use_cdn'] = False

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
        **extra_kwargs)

    if do_resize_token_embedding:
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
