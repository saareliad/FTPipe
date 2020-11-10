import functools

from models.normal.cep import Net
from pipe.models.registery.model_handler import CommonModelHandler, register_model


def register_cep_model(n=50, k=11, c=500, n_split=4):
    model = Net(n, c, n_split=n_split)
    return model


class CEPModelHandler(CommonModelHandler):
    def __init__(self, normal_model_fn, *args, **kw):
        super().__init__(*args, **kw)
        self.normal_model_fn = normal_model_fn

    def get_normal_model_instance(self, *args, **kwargs):
        if self.normal_model_instance is None:
            self.normal_model_instance = self.normal_model_fn(*args, **kwargs)
        return self.normal_model_instance


register_model(name="cep_netN50_C500_4p_bw12_metis",
               handler=CEPModelHandler(normal_model_fn=functools.partial(register_cep_model, n=50, c=500, n_split=4),
                                       generated_file_name_or_path="cep_netN50_C500_4p_bw12_metis"))
