from .transformers_cfg import MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS, MODEL_TYPES


def resize_token_embeddings(model, tokenizer):
    # NOTE: must use the same tokenizer used at partitioning.
    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))


def get_block_size(tokenizer, block_size=-1):
    if block_size <= 0:
        # Our input block size will be the max possible for the model
        block_size = tokenizer.max_len_single_sentence
    block_size = min(block_size, tokenizer.max_len_single_sentence)
    return block_size


def pretrained_model_config_and_tokenizer(
        model_type: str,
        model_name_or_path: str,
        config_name: str = "",
        tokenizer_name: str = "",
        do_lower_case: bool = True,
        cache_dir: str = "",
        output_past=False,
        stateless_tied=False,
        **config_kw
):

    config_class, model_class, tokenizer_class = MODEL_TYPES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None,
        **config_kw
        # num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        )

    config.output_past = output_past  # FIXME: deprecated
    # for k, v in config_kw.items():
    #     setattr(config, k, v)

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None)

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None)

    resize_token_embeddings(model, tokenizer)

    if stateless_tied:
        model_to_resize = model.module if hasattr(model, 'module') else model
        model_to_resize.make_stateless_after_loaded_tied_and_resized()

    return model, tokenizer, config


def get_model_tokenizer_and_config_by_name(name):
    cfg = MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.get(name)()

    model, tokenizer, config = pretrained_model_config_and_tokenizer(**cfg)

    return model, tokenizer, config
