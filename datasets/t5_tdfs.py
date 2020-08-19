import t5

def get_dataset(*args, **kw):
    return t5.models.hf_model.get_dataset(*args, **kw)