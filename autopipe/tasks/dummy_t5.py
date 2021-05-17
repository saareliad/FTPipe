import torch

from pipe.models.transformers_utils import resize_token_embeddings
from . import register_task
from .new_t5 import T5Partitioner, ParsePartitioningT5Opts, TiedT5ForConditionalGeneration, T5Config, T5Tokenizer


class DumT5Partitioner(T5Partitioner):

    def get_model(self, args) -> torch.nn.Module:
        # base = not args.lmhead
        # tied = args.stateless_tied
        #
        # model_cls = TiedT5ForConditionalGeneration
        #
        # config_cls = T5Config
        # tokenizer_class = T5Tokenizer

        explicitly_set_dict = {
            "return_dict": False,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_only": True,
            "precomputed_masks": args.precompute_masks,

        }

        config_class = T5Config
        config = config_class.from_pretrained("t5-11b")

        config.num_layers = 1
        config.num_decoder_layers = 1

        for k, v in explicitly_set_dict.items():
            setattr(config, k, v)

        tokenizer = T5Tokenizer.from_pretrained("t5-11b")

        self.tokenizer = tokenizer
        self.config = config

        model = TiedT5ForConditionalGeneration(config)
        do_resize_token_embedding = True
        stateless_tied = True

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
                raise ValueError(f"Problem making model stateless. ") #model_name_or_path: {model_name_or_path}")



        return model


    def get_input(self, args, analysis=False):
        batch  = super().get_input(args, analysis=analysis)

        if  self.config.num_decoder_layers == 0:
            del batch['decoder_input_ids']
            del batch['decoder_attention_mask']

        return batch
            # batch = {
            #     'input_ids': input_ids,
            #     "attention_mask": attention_mask,
            #     'decoder_input_ids': decoder_input_ids,
            #     "decoder_attention_mask": decoder_attention_mask,
            #     'labels': lm_labels,
            # }


register_task("dummy_t5", ParsePartitioningT5Opts, DumT5Partitioner)
