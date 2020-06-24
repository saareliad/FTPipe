import pandas as pd
import json
import glob
import os
import argparse
# TODO: result dir


def alg(fn):
    for a in ["msnag", "aggmsnag", "stale"]:
        ga_alg = f"{a}_ws_ga"
        if ga_alg in fn:
            return ga_alg
        ws_alg = "{a}_ws"
        if ws_alg+"_" in fn:
            return ws_alg

    return "gpipe" if "gpipe" in fn else "aggmsnag" if "aggmsnag" in fn else "msnag" if "msnag" in fn else "stale" if "stale" in fn else "BUG"


def read_desc_df(path="desc.csv"):
    df = pd.read_csv(path,
                     index_col=[0, 1, 2],
                     header=[0, 1],
                     skipinitialspace=True)
    return df


def filter_desc_df_squad(desc):
    df = desc
    return df[[(i, j) for i in ['f1', 'em', 'total_time']
               for j in ['mean', 'max', 'min', 'std']]]


def filter_desc_df_lm(desc):
    df = desc
    return df[[(i, j) for i in ['ppl', 'total_time']
               for j in ['mean', 'max', 'min', 'std']]]


def write_squad_desc_df():
    ls = glob.glob
    records = []
    for f in ls("*.json"):
        d = {}
        with open(f, "rb") as fd:
            r = json.load(fd)
            d['name'] = f
            d['alg'] = alg(f)
            d['seed'] = r['config']['seed']
            d['agg'] = r['config']['step_every']
            agg = d['agg']
            mb = r['config']['bs_train']
            d['batch'] = agg * mb
            d['total_time'] = r['config']['exp_total_time']
            d['f1'] = r['results']['squad_results']['2']['f1']
            d['em'] = r['results']['squad_results']['2']['exact']

        records.append(d)

    df = pd.DataFrame.from_records(records)
    print(df)
    desc = df.groupby(['alg', 'batch', 'agg']).describe()
    desc = desc[['f1', 'em', 'total_time']]
    print(desc)

    # TODO: to csv with headers
    desc.to_csv("desc.csv", index=True)
    df.to_csv("df.csv", index=False)


def write_lm_desc_df():
    ls = glob.glob
    records = []
    for f in ls("*.json"):
        d = {}
        with open(f, "rb") as fd:
            r = json.load(fd)
            d['name'] = f
            d['alg'] = alg(f)
            d['seed'] = r['config']['seed']
            d['agg'] = r['config']['step_every']
            agg = d['agg']
            mb = r['config']['bs_train']
            d['batch'] = agg * mb
            d['total_time'] = r['config']['exp_total_time']
            d['ppl'] = r['results']['test_ppl'][-1]

        records.append(d)

    df = pd.DataFrame.from_records(records)
    print(df)
    desc = df.groupby(['alg', 'batch', 'agg']).describe()
    desc = desc[['ppl', 'total_time']]
    print(desc)

    # TODO: to csv with headers
    desc.to_csv("desc.csv", index=True)
    df.to_csv("df.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Squad results analysis')
    parser.add_argument("--task",
                        choices=["squad", "lm"],
                        default="squad",
                        help="Learning Task")

    parser.add_argument('--write', action='store_true', default=False)
    parser.add_argument('--read', action='store_true', default=False)
    parser.add_argument('--filter', action='store_true', default=False)

    args = parser.parse_args()

    WRITE = {'squad': write_squad_desc_df, 'lm': write_lm_desc_df}
    FILTER = {'squad': filter_desc_df_squad, 'lm': filter_desc_df_lm}


    if args.write:
        WRITE[args.task]()

    if args.read:
        desc = read_desc_df()
        if args.filter:
            df = FILTER[args.task](desc)
            print(df)
        else:
            print(desc)

    # write_squad_desc_df()