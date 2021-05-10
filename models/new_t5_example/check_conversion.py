from modeling_t5 import T5Model as TiedT5Model
from modeling_t5 import T5ForConditionalGeneration as TiedT5ForConditionalGeneration
from transformers import T5Model, T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch

# run from current dir.
if __name__ == '__main__':

    def normal_model_check():
        model = TiedT5Model.from_pretrained("t5-small")
        # config = config_class
        model.make_stateless()
        output = model(**model.dummy_inputs)

        hfmodel = T5Model.from_pretrained("t5-small")
        output2 = hfmodel(**hfmodel.dummy_inputs)

        for i in output2:
            v1 = output[i]
            v2 = output2[i]
            if isinstance(v2, torch.Tensor):
                print("checking output")
                assert torch.allclose(v1, v2)

        print("OK")


    def generation_model_check():
        config = T5Config.from_pretrained("t5-small")
        config.output_only = True
        config.use_cache = False
        config.precomputed_masks = False
        config.return_dict = False
        config.output_attentions = False
        config.output_scores = False
        config.output_hidden_states = False
        model = TiedT5ForConditionalGeneration.from_pretrained("t5-small",config=config)

        # config = config_class

        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>',
                                return_tensors='pt').input_ids


        dummy_inputs = dict(input_ids=input_ids, labels=labels)
        model.make_stateless()
        output = model(**dummy_inputs)
        if not isinstance(output, tuple):
            output = output,

        hfmodel = T5ForConditionalGeneration.from_pretrained("t5-small",config=config)

        output2 = hfmodel(**dummy_inputs)
        output2 = output2[0],

        if isinstance(output2, dict):
            raise NotImplementedError()

        assert len(output) == len(output2), (len(output), len(output2)) #output, output2)
        for i in range(len(output2)):
            v1 = output[i]
            v2 = output2[i]
            if isinstance(v2, torch.Tensor):
                print("checking output")
                assert torch.allclose(v1, v2)

        print("OK")


    #normal_model_check()
    generation_model_check()

    # print(output)