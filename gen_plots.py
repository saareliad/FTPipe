from misc import plot
from experiments import load_experiment
import matplotlib.pyplot as plt
import os


def add_plot(fn, legened, fig=None):
    config, fit_res = load_experiment(fn)
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot.plot_fit(fit_res, fig=fig, log_loss=False,
                            legend=legened, loss_per_batch=loss_per_batch)
    return fig, ax


def current_status():
    fig = None
    fig, ax = add_plot("results/stale_weights_ninja4.json",
                       "ninja_stale_weights", fig=fig)
    fig, ax = add_plot(
        "results/ninja_msnag_clone_chunks_after_fix.json", "ninja_msnag", fig=fig)

    # fig, ax = add_plot("results/stale_weights_rambo2_1.json", "rambo_stale_weights", fig=fig)
    fig, ax = add_plot("results/stale_weights_rambo2.json",
                       "rambo_stale_weights", fig=fig)  # Newer than above

    fig, ax = add_plot(
        "results/rambo2_msnag_clone_chunks_after_fix.json", "rambo_msnag", fig=fig)

    # Memory efficient msnag
    fig, ax = add_plot("results/ninja_msnag_calc.json",
                       "ninja_msnag_calc", fig=fig)

    # Gap aware:
    fig, ax = add_plot("results/dummy_ga.json", "gap_aware", fig=fig)

    gen_plot(out_dir='results', out_base_name='current_status.png')


def gap_aware():
    fig = None
    # Gap aware:
    fig, ax = add_plot("results/dummy_ga.json", "gap_aware", fig=fig)
    fig, ax = add_plot("results/dummy_ga_no_wd.json", "gap_aware_no_wd", fig=fig)

    gen_plot(out_dir='results', out_base_name='gap_aware.png')


def gen_plot(out_dir='results', out_base_name='current_status.png'):
    plt.plot()
    out_file_name = os.path.join(out_dir, out_base_name)
    plt.savefig(out_file_name)
    print(f"-I- Generated: \"{out_file_name}\"")


def four_partitions():
    fig = None
    fig, ax = add_plot("results/wrn_16x4_p4_stale_weights_1.json", "4p_stale_5_warmup", fig=fig)
    fig, ax = add_plot("results/wrn_16x4_p4_stale_weights_15_warmup.json", "4p_stale_15_warmup", fig=fig)

    # TODO: add after run done, change name to json json
    # fig, ax = add_plot("results/wrn_16x4_p4_stale_weights_35_warmup.json", "4p_stale_35_warmup", fig=fig)

    gen_plot(out_dir='results', out_base_name='4p.png')


if __name__ == "__main__":
    # current_status()
    # gap_aware()
    four_partitions()
    exit(0)
