import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from all_results_to_df import print_uniques


def p1(graph='test_acc'):
    csv = "4partitions.csv"
    out_file_name = f"{graph}.png"
    out_file_name = os.path.join(".", out_file_name)

    df = pd.read_csv(csv).query("dataset == 'cifar100'")
    ax = sns.lineplot(x="epoch", y=graph, hue="alg", data=df)

    ax.set_title(graph)

    model = pd.unique(df.model)
    assert(len(model) == 1)
    model = model[0]

    fig = ax.get_figure()
    fig.savefig(out_file_name)

    print(f"saving file to {out_file_name}")


def p1_fit_plots():
    for graph in ['test_acc', 'train_acc', 'train_loss', 'test_loss']:
        plt.figure()
        p1(graph)


def p2():
    csv = "4partitions.csv"
    out_file_name = "output.png"
    out_file_name = os.path.join(".", out_file_name)

    df = pd.read_csv(csv).query("dataset == 'cifar100'").query("epoch == 200")
    ax = sns.barplot(x="epoch", y="test_acc", hue="alg", data=df)

    model = pd.unique(df.model)
    assert(len(model) == 1)
    model = model[0]

    ax.set_ylim(80, 83)
    ax.set_title(model)
    fig = ax.get_figure()
    fig.savefig(out_file_name)
    print(f"saving file to {out_file_name}")


def p2_2partitions(model='wrn_28x10_c100_dr03_p2'):
    csv = "2partitions.csv"
    out_file_name = f"{model}_output.png"
    out_file_name = os.path.join(".", out_file_name)

    # model = 'wrn_28x10_c100_dr03_p2'
    df = pd.read_csv(csv).query(
        "dataset == 'cifar100' and model == @model").query("epoch == 200")
    ax = sns.barplot(x="epoch", y="test_acc", hue="alg", data=df)

    model = pd.unique(df.model)
    assert(len(model) == 1)
    model = model[0]

    ax.set_ylim(80, 83)
    ax.set_title(model)
    fig = ax.get_figure()
    fig.savefig(out_file_name)
    print(f"saving file to {out_file_name}")


def p2_2partitions_16x4(model='wrn_16x4_c100_p2'):
    csv = "2partitions.csv"
    out_file_name = f"{model}_output.png"
    out_file_name = os.path.join(".", out_file_name)

    # model = 'wrn_28x10_c100_dr03_p2'
    df = pd.read_csv(csv).query(
        "dataset == 'cifar100' and model == @model").query("epoch == 200")
    ax = sns.barplot(x="epoch", y="test_acc", hue="alg", data=df)

    model = pd.unique(df.model)
    assert(len(model) == 1)
    model = model[0]

    ax.set_ylim(75, 78)
    ax.set_title(model)
    fig = ax.get_figure()
    fig.savefig(out_file_name)
    print(f"saving file to {out_file_name}")


def p2_2partitions_all_models():
    for model in ['wrn_16x4_c100_p2', 'wrn_28x10_c100_dr03_p2']:
        plt.figure()
        p2_2partitions(model)
        # plt.clf()


def p3():
    csv = "4partitions.csv"
    out_file_name = "output2.png"
    out_file_name = os.path.join(".", out_file_name)

    df = pd.read_csv(csv).query("dataset == 'cifar10'").query("epoch == 200")
    ax = sns.barplot(x="epoch", y="test_acc", hue="alg", data=df)

    model = pd.unique(df.model)
    assert(len(model) == 1)
    model = model[0]

    ax.set_ylim(94, 96)
    ax.set_title(model)
    fig = ax.get_figure()
    fig.savefig(out_file_name)

    print(f"saving file to {out_file_name}")

def wrn16x4_c100():
    """
    all_results_to_df.py
    -I- There are 236 json files in ['results/4partitions', 'results/sequential']
    -I- Creating....
    -I- Created df.shape: (68370, 23)
    -I- Writing csv: ./4p_seq_ddpsim.csv
    -I- Done
    -I- Describing csv: ./4p_seq_ddpsim.csv
    -I- Analyzed cols: ['alg', 'bs_train', 'model', 'dataset', 'seed', 'step_every']
    -I- length_uniques:
    {'alg': 9, 'bs_train': 4, 'model': 3, 'dataset': 2, 'seed': 5, 'step_every': 4}
    -I- uniques:
    {'alg': array(['stale_ws', 'wp_ga_ws', 'stale', 'wp', 'wp_ga', 'wp_ws',
        'stale_ga_ws', 'stale_ws_pipedream', 'sync'], dtype=object), 'bs_train': array([ 128, 1024,  512,   32]), 'model': array(['wrn_16x4_p4', 'wrn_28x10_c100_dr03_p4', 'wrn_16x4_c100_p4'],
        dtype=object), 'dataset': array(['cifar10', 'cifar100'], dtype=object), 'seed': array([ 1322019, 20202020,   314159,       42, 77777777]), 'step_every': array([1, 2, 4, 8])}
    """
    graph = 'test_acc'
    dataset = 'cifar100'
    model = 'wrn_16x4_c100_p4'

    # 'train_acc'
    csv = "4p_seq_ddpsim.csv"
    out_file_name = f"{graph}_{model}.png"
    out_file_name = os.path.join(".", out_file_name)

    df = pd.read_csv(csv).query("dataset == @dataset and model == @model and bs_train == 128")  # .query("epoch == 200")
    # ax = sns.barplot(x="epoch", y=graph, hue="alg", style='step_every', data=df.query("epoch == 200"))
    ax = sns.catplot(x="epoch", y=graph, hue="alg", col="step_every", kind="bar", data=df.query("epoch == 200"))
    ax.set(ylim=(74, 77))
    
    # ax = sns.lineplot(x="epoch", y=graph, hue="alg", style='step_every', data=df)

    if hasattr(ax, 'get_figure'):
        ax.set_title(f"{graph}_{model}")
        fig = ax.get_figure()
        fig.savefig(out_file_name)
    else:
        # 'FacetGrid'
        ax.savefig(out_file_name)

    print(f"saving file to {out_file_name}")



def wrn16x4_c100_gap():
    """
    all_results_to_df.py
    -I- There are 236 json files in ['results/4partitions', 'results/sequential']
    -I- Creating....
    -I- Created df.shape: (68370, 23)
    -I- Writing csv: ./4p_seq_ddpsim.csv
    -I- Done
    -I- Describing csv: ./4p_seq_ddpsim.csv
    -I- Analyzed cols: ['alg', 'bs_train', 'model', 'dataset', 'seed', 'step_every']
    -I- length_uniques:
    {'alg': 9, 'bs_train': 4, 'model': 3, 'dataset': 2, 'seed': 5, 'step_every': 4}
    -I- uniques:
    {'alg': array(['stale_ws', 'wp_ga_ws', 'stale', 'wp', 'wp_ga', 'wp_ws',
        'stale_ga_ws', 'stale_ws_pipedream', 'sync'], dtype=object), 'bs_train': array([ 128, 1024,  512,   32]), 'model': array(['wrn_16x4_p4', 'wrn_28x10_c100_dr03_p4', 'wrn_16x4_c100_p4'],
        dtype=object), 'dataset': array(['cifar10', 'cifar100'], dtype=object), 'seed': array([ 1322019, 20202020,   314159,       42, 77777777]), 'step_every': array([1, 2, 4, 8])}
    """
    graph = 'p0_gap'
    dataset = 'cifar100'
    model = 'wrn_16x4_c100_p4'
    alg = 'wp_ga_ws'

    # 'train_acc'
    csv = "4p_seq_ddpsim.csv"
    out_file_name = f"{graph}_{model}.png"
    out_file_name = os.path.join(".", out_file_name)

    df = pd.read_csv(csv).query("dataset == @dataset and model == @model and bs_train == 128 and alg== @alg")  # .query("epoch == 200")
    # # ax = sns.barplot(x="epoch", y=graph, hue="alg", style='step_every', data=df.query("epoch == 200"))
    # ax = sns.catplot(x="epoch", y=graph, hue="alg", col="step_every", kind="bar", data=df.query("epoch == 200"))
    # ax.set(ylim=(74, 77))
    
    ax = sns.PairGrid(y_vars=[f'p{i}_gap' for i in range(3)], x_vars=["epoch"], hue='step_every', data=df).map(plt.plot).add_legend()

    
    # ax = sns.lineplot(x="epoch", y=graph, hue="alg", style='step_every', data=df)

    if hasattr(ax, 'get_figure'):
        ax.set_title(f"{graph}_{model}")
        fig = ax.get_figure()
        fig.savefig(out_file_name)
    else:
        # 'FacetGrid'
        ax.savefig(out_file_name)

    print(f"saving file to {out_file_name}")


if __name__ == "__main__":
    

    # plt.clf()
    # p2()
    # p1_fit_plots()
    # wrn16x4_c100()
    wrn16x4_c100_gap()
    # p2_2partitions()
    # p2_2partitions_16x4()
    # p2_2partitions_all_models()

    # p1()

    # p3()
