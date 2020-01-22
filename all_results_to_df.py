import os
from experiments import load_experiment

from typing import NamedTuple  # for bwd compatibility
import itertools
import pandas as pd
import numpy as np
import os
import argparse


def is_json(fn):
    return ".json" in fn


def all_files(path):
    file_names = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if is_json(name):
                fn = (os.path.join(root, name))
                file_names.append(fn)
    return file_names


class InferStuff:
    def __init__(self, config, fit_res):
        self.config = config
        self.fit_res = fit_res
        self.interesting_from_config = {
            i: config[i] for i in ["model", "dataset", 'seed']
        }
        self.all_data = {}
        self.infer_experiment_names()
        self.max_len = 0

    def fix_model_name(self):
        pass

    def infer_experiment_names(self):
        wp = "weight_prediction" in self.config
        ga = "gap_aware" in self.config
        ws = "weight_stashing" in self.config and self.config["weight_stashing"]

        wp_name = "wp" if wp else "stale"
        ga_name = "ga" if ga else ''
        ws_name = "ws" if ws else ''

        names = filter(None, [wp_name, ga_name, ws_name])

        # Alg is contour name
        alg = "_".join(names)

        # Add to dict
        to_add = dict(wp=wp, ga=ga, ws=ws, alg=alg)
        self.interesting_from_config = {
            **self.interesting_from_config, **to_add}

    def merge(self, new_data):
        to_delete = {}
        for i, v in new_data.items():
            # Merge existing stuff
            if i in self.all_data:
                to_delete.add(i)
                assert(i == "epoch")
                if len(v) > len(self.all_data[i]):
                    self.all_data[i] = v
                    self.max_len = len(v)

        # Add non exising stuff
        d = {}
        for i, v in new_data.items():
            if i not in to_delete:
                d[i] = v

        self.all_data = {**self.all_data, **d}

    def fix_gap_for_last_p(self, attr, data, length, gaps):
        if 'gap' in attr and len(data) == 0:
            l_index = f"p{len(gaps) - 1}"
            if l_index in attr:
                return np.zeros(length)

    def infer_epoch_attrs(self):
        attrs = []
        p = itertools.product(['train', 'test'], ['loss', 'acc'])
        traintestlossacc = [
            f'{traintest}_{lossacc}' for (traintest, lossacc) in p]
        gaps = [key for key in self.fit_res.keys() if "gap" in key]
        norms = [key for key in self.fit_res.keys() if "grad_norm" in key]

        attrs.extend(traintestlossacc)
        attrs.extend(gaps)
        attrs.extend(norms)

        attrs = [attr for attr in attrs if (attr in self.fit_res)]

        all_data = {}
        length = None
        for attr in attrs:
            if isinstance(self.fit_res, NamedTuple):
                data = getattr(self.fit_res, attr)
            else:
                data = self.fit_res[attr]

            if not length:
                # Add epoch indicator
                length = len(data)
                self.max_len = length
                all_data['epoch'] = np.arange(1, len(data) + 1)
            else:
                if (len(data) != length):
                    new_data = self.fix_gap_for_last_p(
                        attr, data, length, gaps)
                    if not (new_data is None):
                        data = new_data
                    else:
                        raise NotImplementedError(
                            f"Supported only for same length: attr:{attr}, len(data):{len(data)} len:{length}")

            # Add the data
            all_data[attr] = data

        # Merge
        self.merge(all_data)

    def replicate(self):
        """ Replicate stuff """
        for i, v in self.interesting_from_config.items():
            self.all_data[i] = [v]*self.max_len

    def to_df(self):
        return pd.DataFrame(self.all_data)


def proccess_file(f):
    """ Returns a dataframe """
    config, fit_res = load_experiment(f)
    inferer = InferStuff(config, fit_res)
    inferer.infer_epoch_attrs()
    inferer.replicate()
    return inferer.to_df()


def all_results_to_csv(root_path, csv_name):
    files = all_files(root_path)
    print(f"-I- There are {len(files)} json files in {path}")
    print("-I- Creating....")
    df = pd.concat([proccess_file(f) for f in files], sort=False)
    print(f"-I- Created df.shape: {df.shape}")
    print(f"-I- Writing csv: {csv_name}")
    df.to_csv(csv_name, index=False)
    print("-I- Done")


if __name__ == "__main__":
    path = "results/2partitions"
    csv_name = "2partitions.csv"
    csv_name = os.path.join(".", csv_name)
    all_results_to_csv(path, csv_name)
