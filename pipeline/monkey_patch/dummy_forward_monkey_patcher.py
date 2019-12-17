from .patch import dummy_forward_monkeypatch
from .find_modules import find_modules


def convert_child(model, b4, after):
    # TODO: write another version which is more efficient, can do this for all models in one pass.
    # (instead of many passes)
    for child_name, child in model.named_children():
        if child is b4:
            setattr(model, child_name, after)
            # print(f"Converted {child_name}")
        else:
            convert_child(child, b4, after)


class DummyForwardMonkeyPatcher:
    def __init__(self, model, classes_list_to_patch):
        """ List of model names to patch """
        self.model = model
        self.classes_to_patch = classes_list_to_patch
        self.models = []
        self.encapsulators = []
        self.fmodels = []

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

    def replace_for_dummy(self):
        for m, fm in zip(self.models, self.fmodels):
            convert_child(self.model, m, fm)

    def replace_for_forward(self):
        for m, fm in zip(self.models, self.fmodels):
            convert_child(self.model, fm, m)

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


if __name__ == "__main__":
    # Unit test...
    # python -m monkey_patch.dummy_forward_monkey_patcher
    test()