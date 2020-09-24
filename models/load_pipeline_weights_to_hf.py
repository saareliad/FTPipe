import torch
import re
import os
import abc
from transformers import AutoModel, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
from .transformers_utils import resize_token_embeddings
from .model_handler import AVAILABLE_MODELS


class Loader(abc.ABC):
    ALLOW_UNSHARED = {}
    ALLOW_UNLOADEDED = {}

    @abc.abstractmethod
    def load_from_saved_pipeline(self, args, to_original=True, **kw):
        # returns: model, extra (dict)
        raise NotImplementedError()

    def _check_load_matching(self, original_state, unified_state):
        # Reruns if loading is strict
        if not (self.ALLOW_UNSHARED or self.ALLOW_UNLOADEDED):
            return True

        not_shared = set(original_state.keys()).difference(
            unified_state.keys())
        problematic = not_shared.difference(self.ALLOW_UNSHARED)
        if problematic:
            raise ValueError(
                f"Parameters {problematic} are not in unified_state, but is in original state"
            )
        # see if enite unified state is loaded
        not_shared = set(unified_state.keys()).difference(
            original_state.keys())
        problematic = not_shared.difference(self.ALLOW_UNLOADEDED)
        if problematic:
            raise ValueError(
                f"Parameters {problematic} are in unified_state, but not in original state"
            )

        return False


def base_checkoint_name(name_prefix, stage):
    return f"{name_prefix}_Partition{stage}.pt"


class HFLoader(Loader):
    IS_HUGGINFACE_TRANSFORMER = True

    def __init__(self, hf_transformers_model_class=AutoModel):
        super().__init__()
        self.MODEL_CLASS = hf_transformers_model_class

    def load_from_saved_pipeline(self, args, to_original=True, **kw):
        cfg = args.model
        partitions_saved_dir = args.checkpoints_save_dir
        name_prefix = getattr(args, "checkpoints_save_name_prefix", "")
        add_to_prefix = kw.pop("add_to_prefix", "")
        name_prefix += add_to_prefix

        # Get unified state dict
        unified_state = self.get_unified_state_dict(cfg, name_prefix,
                                                    partitions_saved_dir)
        print(f"-I- Loaded state dict from {partitions_saved_dir}")

        if to_original:
            # Load state dict to original model
            unified_state = self.substitue_state_dict_keys_back_to_original(
                unified_state)
            model_name_or_path = args.model_name_or_path
            # TODO: call with other hyperparameters...

            if all([k in kw for k in ["model", "tokenizer", "config"]]):
                model = kw.get("model")
                tokenizer = kw.get("tokenizer")
                config = kw.get("config")
            else:
                model, tokenizer, config = self.get_hf_original_model_tokenizer_and_config(
                    model_name_or_path)
        else:
            # load the model used for training/finetuning.
            # generated = get_generated_module(cfg)
            if all([k in kw for k in ["model", "tokenizer", "config"]]):
                model = kw.get("model")
                tokenizer = kw.get("tokenizer")
                config = kw.get("config")
            else:
                handler = AVAILABLE_MODELS.get(cfg)
                model = handler.get_normal_model_instance()
                tokenizer = handler.tokenizer
                config = handler.config

        strict = self._check_load_matching(
            original_state=model.state_dict(), unified_state=unified_state)
        model.load_state_dict(unified_state, strict=strict)
        print("-I- Loaded state into the model")

        extra = dict(tokenizer=tokenizer, config=config)
        return model, extra

    def get_unified_state_dict(self, cfg, name_prefix, partitions_saved_dir):
        n_stages = AVAILABLE_MODELS.get(cfg).get_pipe_config().n_stages
        names = [base_checkoint_name(name_prefix, stage=i) for i in range(n_stages)]
        names = [os.path.join(partitions_saved_dir, name) for name in names]
        print(f"-V- loading from {names}")
        loaded = [torch.load(name, map_location="cpu") for name in names]
        unified_state = dict()
        for d in loaded:
            unified_state.update(d)
        return unified_state

    def get_hf_original_model_tokenizer_and_config(
            self,
            model_name_or_path,
            cache_dir="",
            config_name=None,
            tokenizer_name=None,
            tokenizer_kw=dict(do_lower_case=False),
            config_kw=dict(),
            resize_embeds=True,
    ):
        """Get Huggingface model, tokenizer and config we want to load to."""
        config, unsed = AutoConfig.from_pretrained(
            config_name if config_name else model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
            return_unused_kwargs=True,
            **config_kw)

        if unsed:
            print(
                f"warning: Unused config kwargs when loading transformer model: {unsed}"
            )

        # TODO: Unused?
        # for k, v in explicitly_set_dict.items():
        #     setattr(config, k, v)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir if cache_dir else None,
            **tokenizer_kw)

        use_cdn = model_name_or_path not in {"t5-11b"}
        model = self.MODEL_CLASS.from_pretrained(
            model_name_or_path,
            from_tf=bool('.ckpt' in model_name_or_path),
            config=config,
            cache_dir=cache_dir if cache_dir else None,
            use_cdn=use_cdn)

        if resize_embeds:
            resize_token_embeddings(model, tokenizer)

        return model, tokenizer, config

    @abc.abstractmethod
    def substitue_state_dict_keys_back_to_original(self, training_state_dict):
        raise NotImplementedError()


# TODO: make it generic...
class T5HFLoader(HFLoader):
    def __init__(self, hf_transformers_model_class=T5ForConditionalGeneration):
        super().__init__(
            hf_transformers_model_class=hf_transformers_model_class)

    def substitue_state_dict_keys_back_to_original(self, training_state_dict):
        # TODO: training_state_dict is origianl state dict used at our training.
        d = dict()
        for k, v in training_state_dict.items():
            # we modified keys from prefix.block.N.layer.M.suffix into prefix.N.M.suffix
            # this regex substitution performs the reverse transformation
            new_key = re.sub(r'([0-9]+.)([0-9]+.)', r'block.\1layer.\2', k)
            d[new_key] = v

        # in case we load weights from the tied model
        if "shared_embed_weight" in d:
            w = d.pop("shared_embed_weight")
            d['shared.weight'] = d['encoder.embed_tokens.weight'] = d[
                'decoder.embed_tokens.weight'] = w
        return d


if __name__ == "__main__":
    import types

    args = dict(model_name_or_path="t5-small",
                checkpoints_save_name_prefix="tst_t5_",
                model="t5_small_tied_lmhead_4p_bw12_async_squad1",
                checkpoints_save_dir="tstloading")
    args = types.SimpleNamespace(**args)
    to_original = True

    cfg = args.model
    handler = AVAILABLE_MODELS.get(cfg)
    generated = handler.get_generated_module()
    model = handler.get_normal_model_instance()
    tokenizer = handler.tokenizer
    config = handler.config

    layers, tensors = handler.get_layers_and_tensors()
    pipe_config = handler.get_pipe_config()

    n_stages = pipe_config.n_stages
    partitions = [
        getattr(generated, f"Partition{i}")(layers, tensors, device='cpu')
        for i in range(n_stages)
    ]
    os.makedirs(args.checkpoints_save_dir, exist_ok=True)
    name_prefix = args.checkpoints_save_name_prefix
    for i, partition in enumerate(partitions):
        fn = os.path.join(args.checkpoints_save_dir, base_checkoint_name(name_prefix, stage=i))
        torch.save(partition.state_dict(), fn)
        print(f"-I- saved to {fn}")

    print("-I- loading")
    # Test loading
    loader = T5HFLoader(hf_transformers_model_class=T5ForConditionalGeneration)
    hugg, extra = loader.load_from_saved_pipeline(args, to_original=True)
    config = extra['config']
    tokenizer = extra['tokenizer']

    print("generating output")
    input_ids = tokenizer.encode("summarize: Hello, my dog is cute",
                                 return_tensors="pt")  # Batch size 1
    outputs = hugg.generate(input_ids)
    print(outputs)
