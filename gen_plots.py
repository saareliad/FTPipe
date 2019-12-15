from misc import plot
from experiments import load_experiment
import matplotlib.pyplot as plt

if __name__ == "__main__":

    fig = None
    # config, fit_res = load_experiment("results/ninja4_stale_weights.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="ninja4_stale weights")

    # config, fit_res = load_experiment("results/ninja4_msnag_clone.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="ninja4_msnag")

    # config, fit_res = load_experiment("results/ninja4_stale_weights_flush_2.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="ninja4_stale_weights_flush2")

    # config, fit_res = load_experiment("results/rambo2_msnag_clone.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_msnag")

    config, fit_res = load_experiment("results/rambo2_stale_weights.json")
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_stale_weights")

    config, fit_res = load_experiment("results/rambo2_msnag_clone_chunks.json")
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_msnag")

    plt.plot()
    plt.savefig('results/rambo2.png')
