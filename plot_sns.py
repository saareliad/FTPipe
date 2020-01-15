import pandas as pd
import seaborn as sns


csv = "4partitions.csv"
out_file_name = "output.png"

df = pd.read_csv(csv).query("dataset == 'cifar100'")
ax = sns.lineplot(x="epoch", y="test_acc", hue="alg", data=df)
fig = ax.get_figure()
fig.savefig(out_file_name)
print(f"saving file to {out_file_name}")
