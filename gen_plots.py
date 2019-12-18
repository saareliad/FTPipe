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

    def add_plot(fn, legened, fig=None):
        config, fit_res = load_experiment(fn)
        loss_per_batch = "loss_per_batch" in config['statistics']
        fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend=legened, loss_per_batch=loss_per_batch)
        return fig, ax

    fig, ax = add_plot("results/stale_weights_ninja4.json", "ninja_stale_weights", fig=fig)
    fig, ax = add_plot("results/ninja_msnag_clone_chunks_after_fix.json", "ninja_msnag", fig=fig)

    fig, ax = add_plot("results/stale_weights_rambo2.json", "rambo_stale_weights", fig=fig)
    fig, ax = add_plot("results/rambo2_msnag_clone_chunks_after_fix.json", "rambo_msnag", fig=fig)

    # ninja_msnag_clone_chunks_after_fix
    
    # config, fit_res = load_experiment("results/stale_weights_ninja4.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="nnja4_stale_weights")

    # config, fit_res = load_experiment("results/rambo2_stale_weights.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_stale_weights")

    # config, fit_res = load_experiment("results/rambo2_msnag_clone_chunks.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_msnag")

    # config, fit_res = load_experiment("results/rambo2_msnag_clone_chunks_after_fix.json")
    # loss_per_batch = "loss_per_batch" in config['statistics']
    # fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False, legend="rambo_msnag_clone_all")

    plt.plot()
    plt.savefig('results/current_status.png')
