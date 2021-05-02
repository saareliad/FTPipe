from modeling_t5 import T5Model as TiedT5Model
from modeling_t5 import T5ForConditionalGeneration as TiedT5ForConditionalGeneration
from transformers import T5Model, T5ForConditionalGeneration
import torch

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
                assert torch.allclose(v1, v2)

        print("OK")


    def generation_model_check():
        model = TiedT5ForConditionalGeneration.from_pretrained("t5-small")
        # config = config_class
        model.make_stateless()
        output = model(**model.dummy_inputs)

        hfmodel = T5ForConditionalGeneration.from_pretrained("t5-small")
        output2 = hfmodel(**hfmodel.dummy_inputs)

        for i in output2:
            v1 = output[i]
            v2 = output2[i]
            if isinstance(v2, torch.Tensor):
                assert torch.allclose(v1, v2)

        print("OK")


    normal_model_check()
    generation_model_check()

    # print(output)