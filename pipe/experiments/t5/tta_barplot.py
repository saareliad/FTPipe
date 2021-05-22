import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


d = [

    {
        'pipeline': 'gpipe',
        'alg': 'mixed',
        'dataset': 'wic',
        'speedup_over_seq': 3.347102
    },

    {
        'pipeline': 'gpipe',
        'alg': 'seq',
        'dataset': 'wic',
        'speedup_over_seq': 6.094234
    },

    {
        'pipeline': 'gpipe',
        'alg': 'mixed',
        'dataset': 'rte',
        'speedup_over_seq': 4.591634
    },

    {
        'pipeline': 'gpipe',
        'alg': 'seq',
        'dataset': 'rte',
        'speedup_over_seq': 7.060724
    },

    {
        'pipeline': 'gpipe',
        'alg': 'mixed',
        'dataset': 'boolq',
        'speedup_over_seq': (3.103546 * 0.7670211892949849)  # see warning above
    },
    {
        'pipeline': 'gpipe',
        'alg': 'seq',
        'dataset': 'boolq',
        'speedup_over_seq': 3.505261
    },

    {
        'pipeline': 'stale',
        'alg': 'mixed',
        'dataset': 'wic',
        'speedup_over_seq': 2.800590  ## TODO
    },
    #     {
    #     'pipeline': 'stale',
    #     'alg': 'seq',
    #     'dataset':'wic',
    #     'speedup_over_seq': 1.54 * 2.800590  ## TODO
    #     },

    {
        'pipeline': 'stale',
        'alg': 'mixed',
        'dataset': 'rte',
        'speedup_over_seq': 2.474085518499304  # FIXME: what is going on here?
    },

    {
        'pipeline': 'stale',
        'alg': 'seq',
        'dataset': 'rte',
        'speedup_over_seq': 5.262030
    },

    {
        'pipeline': 'stale',
        'alg': 'mixed',
        'dataset': 'boolq',
        'speedup_over_seq': 1.847583
    },

    {
        'pipeline': 'stale',
        'alg': 'seq',
        'dataset': 'boolq',
        'speedup_over_seq': 3.061177
    }
]
palette = sns.color_palette("dark")
# palette = [palette[1], palette[0]]
df = pd.DataFrame.from_records(d)
print("option 1")
# sns.set_theme(style="white")
set_style()
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="dataset", y="speedup_over_seq", hue="alg", col="pipeline",  # hue="dataset",
    ci="sd", palette=palette, height=4, alpha=.6,  # aspect=.7, # alpha=.6, height=6
    #     col_order=["boolq", "wic", "rte"],
    hue_order=['seq', 'mixed'],
    legend=False,
)
axes = np.array(g.axes.flat)
sns.despine(ax=axes[1], left=True)
axes[0].set_xlabel("GPipe")
axes[1].set_xlabel("FTPipe")
# sns.despine(ax=axes[2], left=True)
# g.despine(left=True)
# g.set_axis_labels("", "Time to Accuracy (Hours)")
axes[0].set_ylabel("Time to Accuracy (Hours)")
axes[0].set_title("")
axes[1].set_title("")

fig = plt.gcf()
width = 7
height = width / 1.618
fig.set_size_inches(width, height)

L = plt.legend(loc='upper right', frameon=False)
L.get_texts()[1].set_text("Mixed-pipe")
L.get_texts()[0].set_text("Seq-pipe")

patches = [axes[0].patches[0 + 3], axes[0].patches[1 + 3], axes[0].patches[2 + 3]]
values = [6.094234 / 3.347102, 7.060724 / 4.591634, 3.505261 / (3.103546 * 0.7670211892949849)]
# values = ["x"+ str(i) for i in values]
# plt.rcParams.update({
#     "text.usetex": True,})

for p, v in zip(patches, values):
    axes[0].annotate(f"{v:.2f}x", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', color='gray', rotation=90, xytext=(0, 20),
                     textcoords='offset points')

patches = [axes[1].patches[1 + 3], axes[1].patches[2 + 3]]
values = [5.262030 / 2.402954, 3.061177 / 1.847583]

for p, v in zip(patches, values):
    axes[1].annotate(f"{v:.2f}x", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', color='gray', rotation=90, xytext=(0, 20),
                     textcoords='offset points')

# , axes[1].patches[1], axes[1].patches[3]

plt.savefig("results/paper_plots/VirtualStages_barplot_TTA.pdf", transparent=False, bbox_inches='tight')
