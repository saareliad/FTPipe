import os
from typing import NamedTuple  # for bwd compatibility
import itertools
import pandas as pd
import numpy as np
import os
import argparse
import sys
sys.path.append("..")
from experiments import load_experiment

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

        stat_to_default = {
            'step_every': 1    
        }

        def get_from_cfg(stat):
            # config.get(i, stat_to_default.get(i,None))
            return config[stat] if stat in config else stat_to_default[stat]
        
        self.interesting_from_config = {
            i: get_from_cfg(i) for i in ["model", "dataset", 'seed', 'bs_train', 'step_every']
        }
        self.all_data = {}
        self.infer_experiment_names()
        self.max_len = 0

    def fix_model_name(self):
        pass

    def infer_num_partitions(self):
        # TODO:
        pass

    def infer_experiment_names(self):
        # TODO: add names for SSGD/sequential
        # this is only for pipeline,
        wp = "weight_prediction" in self.config
        ga = "gap_aware" in self.config
        ws = "weight_stashing" in self.config and self.config["weight_stashing"]
        pipedream = "work_scheduler" in self.config and (
            self.config["work_scheduler"] == "PIPEDREAM")
        sync = "is_sync" in self.config and self.config['is_sync']
        ddp = "ddp" in self.config and self.config['ddp']

        wp_name = "wp" if wp else ("stale" if not sync else '')
        ga_name = "ga" if ga else ''
        ws_name = "ws" if ws else ''
        pipedream_name = 'pipedream' if pipedream else ''
        sync_name = 'sync' if sync else ''
        ddp = 'ddp' if ddp else ''
        if ddp:
            sync_name = ''

        names = filter(None, [wp_name, ga_name, ws_name,
                              pipedream_name, sync_name, ddp])

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


def process_file(f):
    """ Returns a dataframe """
    config, fit_res = load_experiment(f)
    inferer = InferStuff(config, fit_res)
    inferer.infer_epoch_attrs()
    inferer.replicate()
    return inferer.to_df()


def all_results_to_csv(root_paths, csv_name):
    if isinstance(root_paths, str):
        root_paths = [root_paths]

    files = []
    for root_path in root_paths:
        files += all_files(root_path)

    # files = [*all_files(root_path) for root_path in root_paths]

    print(f"-I- There are {len(files)} json files in {root_paths}")
    print("-I- Creating....")
    df = pd.concat([process_file(f) for f in files], sort=False)
    print(f"-I- Created df.shape: {df.shape}")
    print(f"-I- Writing csv: {csv_name}")
    df.to_csv(csv_name, index=False)
    print("-I- Done")


## Analysis tools:
def print_uniques(csv, cols=["alg", 'bs_train', "model", "dataset", 'seed', 'step_every']):
    # TODO: args.bs_train * args.step_every
    # TODO: number of partitions
    df = pd.read_csv(csv)
    var_to_uniques = {var: pd.unique(df[var]) for var in cols}
    var_to_len_uniques = {i: len(v) for i, v in var_to_uniques.items()}

    print(f"-I- Describing csv: {csv}")
    print(f"-I- Analyzed cols: {cols}")

    print("-I- length_uniques:")
    print(var_to_len_uniques)

    print("-I- uniques:")
    print(var_to_uniques)


if __name__ == "__main__":

    def two_partitions():
        path = "results/2partitions"
        csv_name = "2partitions.csv"
        csv_out_dir = "."
        csv_name = os.path.join(csv_out_dir, csv_name)
        all_results_to_csv(path, csv_name)

    def four_partitions():
        path = "results/4partitions"
        csv_name = "4partitions.csv"
        csv_out_dir = "."
        csv_name = os.path.join(csv_out_dir, csv_name)
        all_results_to_csv(path, csv_name)

    def all_results_with_sequential():
        paths = [# "results/2partitions",
                 "results/4partitions", 'results/sequential']
        # csv_name = "2p_4p_seq_ddpsim"
        csv_name = "4p_seq_ddpsim.csv"
        csv_out_dir = "."
        csv_name = os.path.join(csv_out_dir, csv_name)
        all_results_to_csv(paths, csv_name)
        print_uniques(csv_name)

    def four_partitions_ddp_for_meeting():
        paths = ["results/4partitions/stale",
        "results/4partitions/ws/",
        "results/4partitions/ws_msnag_ga/",
        "results/4partitions/msnag_ws/",
        "results/4partitions/msnag/",
        "results/4partitions/msnag_ga/",
         'results/ddp_all']     
        csv_name = "for_meeting.csv"
        csv_out_dir = "."
        csv_name = os.path.join(csv_out_dir, csv_name)
        all_results_to_csv(paths, csv_name)
        print_uniques(csv_name)
    
    four_partitions_ddp_for_meeting()
    # all_results_with_sequential()
