import os
from torch import tensor  # for parsing.
import warnings
from pprint import pprint


# def get_epoch_to_ppl():
relative_dir_names = [f"epoch_{i}" for i in range(3)] + [""]
dirs = [os.path.join(os.getcwd(), d) for d in relative_dir_names]
files = [os.path.join(d, "eval_results.txt") for d in dirs]

epoch_to_ppl = {}
for epoch, file in enumerate(files):
    with open(file, "r") as f:
        perplexity = None
        exec(f.read())
        if perplexity is None:
            pass
            # s = "perplexity is None, epoch:{epoch}".format(epoch=epoch)
            # warnings.warn(s)
        else:
            epoch_to_ppl[epoch] = perplexity.item()
# return epoch_to_ppl


# if __name__ == "__main__":
# epoch_to_ppl = get_epoch_to_ppl()
pprint(epoch_to_ppl)
