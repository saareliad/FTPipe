from .patch import dummy_forward_monkeypatch
from .find_modules import find_modules


def convert_child_by_dict(model, dict_id_b4_to_after):
    if not dict_id_b4_to_after:
        return
    for child_name, child in model.named_children():
        if id(child) in dict_id_b4_to_after:
            # print(f"Converted {child_name} to {dict_id_b4_to_after[id(child)]}")
            setattr(model, child_name, dict_id_b4_to_after[id(child)])
        else:
            convert_child_by_dict(child, dict_id_b4_to_after)


class DummyForwardMonkeyPatcher:
    def __init__(self, model, classes_list_to_patch):
        """ List of model names to patch """
        self.model = model
        self.classes_to_patch = classes_list_to_patch
        self.models = []
        self.encapsulators = []
        self.fmodels = []
        self.state_is_dummy = False

        for model_to_patch in self.classes_to_patch:
            found = []  # list of tuples: (access_string, model)
            find_modules(model, "", model_to_patch, found)

            # ACCESS_STRS = [i[0][1:] for i in found]
            self.models += [i[1] for i in found]

            # list of tuples (fmodel, encapsulator)
            monkey_patched_enc_tuples = [dummy_forward_monkeypatch(
                orig_model) for orig_model in self.models]

            self.fmodels += [i[0] for i in monkey_patched_enc_tuples]
            self.encapsulators += [i[1] for i in monkey_patched_enc_tuples]

        # Create dicts.
        # Warning: if id changes - this won't work.
        self.id_models_to_fmodels = {id(m): fm for m, fm in zip(self.models, self.fmodels)}
        self.id_fmodels_to_models = {id(fm): m for m, fm in zip(self.models, self.fmodels)}

    def replace_for_dummy(self):
        if not self.state_is_dummy:
            convert_child_by_dict(self.model, self.id_models_to_fmodels)
            self.state_is_dummy = True

    def replace_for_forward(self):
        if self.state_is_dummy:
            convert_child_by_dict(self.model, self.id_fmodels_to_models)
            self.state_is_dummy = False

    def sync(self):
        for encapsulator, fmodule, module in zip(self.encapsulators, self.fmodels, self.models):
            encapsulator(fmodule, module)


def test():
    import torch
    features = 3
    batch = 3
    model = torch.nn.Sequential(torch.nn.Linear(
        features, features), torch.nn.BatchNorm1d(features))

    patcher = DummyForwardMonkeyPatcher(
        model, classes_list_to_patch=[torch.nn.BatchNorm1d])
    patcher.sync()
    print(model)
    patcher.replace_for_dummy()
    model(torch.randn(features, features))
    print(model)
    print(model[1].state_dict())
    patcher.replace_for_forward()
    print(model)
    print(model[1].state_dict())

    print()
    print("now updating")

    y_pred = model(torch.randn(batch, features))
    print(model)
    print(model[1].state_dict())

    patcher.sync()
    patcher.replace_for_dummy()
    print(model)
    print(model[1].state_dict())
    patcher.replace_for_forward()

    print('-' * 89)
    print("now with grad")
    # Testing parameters replacement clone
    loss_fn = torch.nn.MSELoss()
    y = torch.randn(batch, features)
    loss_fn(y_pred, y).backward()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9)
    optimizer.step()

    print(model)
    print(model[1].state_dict())

    patcher.replace_for_dummy()

    print(model)
    print(model[1].state_dict())


def test_no_grad_and_bwd():
    import torch
    features = 3
    batch = 3
    model = torch.nn.Sequential(torch.nn.Linear(
        features, features), torch.nn.BatchNorm1d(features))

    patcher = DummyForwardMonkeyPatcher(
        model, classes_list_to_patch=[torch.nn.BatchNorm1d])
    patcher.sync()
    patcher.replace_for_dummy()

    with torch.no_grad():
        res = model(torch.randn(batch, features))

    patcher.replace_for_forward()
    torch.nn.functional.mse_loss(model(torch.randn(batch, features)), torch.randn(batch, features)).backward()


if __name__ == "__main__":
    # Unit test.
    # From pipeline dir:
    # python -m monkey_patch.dummy_forward_monkey_patcher
    test()
    # test_no_grad_and_bwd()