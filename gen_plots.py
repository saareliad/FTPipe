from misc import plot
from experiments import load_experiment
import matplotlib.pyplot as plt

if __name__ == "__main__":

    config, fit_res = load_experiment("results/stale_weights.json")
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot.plot_fit(fit_res, fig=None, log_loss=False, legend="stale weights")
    plt.plot()
    plt.savefig('results/stale_weights.png')
