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


if __name__ == "__main__":

    # plt.clf()
    # p2()
    p1_fit_plots()
    # p2_2partitions()
    # p2_2partitions_16x4()
    # p2_2partitions_all_models()

    # p1()

    # p3()
