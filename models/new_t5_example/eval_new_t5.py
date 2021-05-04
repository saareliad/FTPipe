# TODO: make it generic...
from pipe.models.load_pipeline_weights_to_hf import HFLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, T5ForConditionalGeneration


class NewT5HFLoader(HFLoader):
    def __init__(self, hf_transformers_model_class=T5ForConditionalGeneration):
        super().__init__(
            hf_transformers_model_class=hf_transformers_model_class)

    def substitue_state_dict_keys_back_to_original(self, training_state_dict):
        # TODO: training_state_dict is origianl state dict used at our training.
        d = dict()

        for k, v in training_state_dict.items():
            # we modified keys from prefix.block.N.layer.M.suffix into prefix.N.M.suffix
            # this regex substitution performs the reverse transformation
            # new_key = re.sub(r'([0-9]+.)([0-9]+.)', r'block.\1layer.\2', k)
            d[k] = v

        # in case we load weights from the tied model
        if "shared_embed_weight" in d:
            w = d.pop("shared_embed_weight")
            d['shared.weight'] = d['encoder.embed_tokens.weight'] = d[
                'decoder.embed_tokens.weight'] = w
        return d
