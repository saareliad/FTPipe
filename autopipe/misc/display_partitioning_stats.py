import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_df(L_to_minmax, L_to_num_stages, L_to_best_objective):
    def list_keys(x):
        return list(x.keys())

    assert list_keys(L_to_num_stages) == list_keys(L_to_best_objective) == list_keys(L_to_minmax)

    records = [
        dict(
            L=L, stages=stages, objective=objective
        )
        for L, stages, objective in zip(L_to_num_stages.keys(), L_to_num_stages.values(), L_to_best_objective.values())

    ]

    df = pd.DataFrame.from_records(records)
    df['objective'] /= 1e4
    return df


def plot_L_to_objective(df):
    sns.barplot(x='L', y='objective', data=df)


def t5_3b():
    L_to_minmax = {8: 6636137.099132873, 16: 5638619.469868817, 24: 4589449.469869904, 32: 4287169.033868238,
                   40: 4103992.787624088, 48: 4155925.9036572957, 56: 4201891.442869065, 64: 4098424.4248143705}
    L_to_num_stages = {8: 8, 16: 15, 24: 23, 32: 31, 40: 35, 48: 45, 56: 46, 64: 61}
    L_to_best_objective = {8: 5529952.12069609, 16: 4083569.301969547, 24: 3890990.6952651218, 32: 3749676.0803660783,
                           40: 3725981.80166127, 48: 3727633.9440387157, 56: 3726344.331668224, 64: 3726255.025884728}
    # run:4 {'correct': 75, 'fp': 7, 'fn': 1, 'mistakes': 8}
    df = get_df(L_to_minmax, L_to_num_stages, L_to_best_objective)
    sns.barplot(x='L', y='objective', data=df)
    print(df)
    plt.show()


def t5_base():
    L_to_minmax = {8: 1134675.3105777686, 16: 849120.8780806304, 24: 941463.7233041492, 32: 804757.1768176018,
                          40: 805675.4889778851, 48: 774942.1858372729, 56: 789406.8785677295, 64: 766349.086868281}
    L_to_num_stages = {8: 8, 16: 15, 24: 23, 32: 27, 40: 33, 48: 45, 56: 48, 64: 60}
    L_to_best_objective ={8: 929510.4466206762, 16: 715666.8662249836, 24: 735406.3966344037, 32: 712613.3374136146, 40: 700121.7308791562,
     48: 695731.4327363339, 56: 695933.99863597, 64: 696800.3580791669}
    # final_countdown_iteration: 18 / 19
    # {'correct': 1, 'fp': 0, 'fn': 18, 'mistakes': 18}
    df = get_df(L_to_minmax, L_to_num_stages, L_to_best_objective)
    sns.barplot(x='L', y='objective', data=df)
    print(df)
    plt.show()

if __name__ == '__main__':
    t5_base()
