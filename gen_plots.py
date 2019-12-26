from misc import plot
from experiments import load_experiment
import matplotlib.pyplot as plt
import os

# plot.plot_grad_norm
# plot.plot_fit


def add_plot(fn, legened, fig=None, plot_fn=plot.plot_fit):
    config, fit_res = load_experiment(fn)
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot_fn(fit_res, fig=fig, log_loss=False,
                      legend=legened, loss_per_batch=loss_per_batch)
    return fig, ax


def current_status():
    plot_fn = plot.plot_fit
    fig = None

    fig, ax = add_plot("results/stale_weights_ninja4.json",
                       "ninja_stale_weights", fig=fig, plot_fn=plot_fn)
    fig, ax = add_plot(
        "results/ninja_msnag_clone_chunks_after_fix.json", "ninja_msnag", fig=fig, plot_fn=plot_fn)

    # fig, ax = add_plot("results/stale_weights_rambo2_1.json", "rambo_stale_weights", fig=fig)
    fig, ax = add_plot("results/stale_weights_rambo2.json",
                       "rambo_stale_weights", fig=fig)  # Newer than above

    fig, ax = add_plot(
        "results/rambo2_msnag_clone_chunks_after_fix.json", "rambo_msnag", fig=fig, plot_fn=plot_fn)

    # Memory efficient msnag
    fig, ax = add_plot("results/ninja_msnag_calc.json",
                       "ninja_msnag_calc", fig=fig)

    # Gap aware:
    fig, ax = add_plot("results/dummy_ga.json", "gap_aware",
                       fig=fig, plot_fn=plot_fn)

    gen_plot(out_dir='results', out_base_name='current_status.png')


def gap_aware():
    plot_fn = plot.plot_fit
    fig = None
    # Gap aware:
    fig, ax = add_plot("results/dummy_ga.json", "gap_aware",
                       fig=fig, plot_fn=plot_fn)
    fig, ax = add_plot("results/dummy_ga_no_wd.json",
                       "gap_aware_no_wd", fig=fig, plot_fn=plot_fn)

    gen_plot(out_dir='results', out_base_name='gap_aware.png')


def gen_plot(out_dir='results', out_base_name='current_status.png'):
    plt.plot()
    out_file_name = os.path.join(out_dir, out_base_name)
    plt.savefig(out_file_name)
    print(f"-I- Generated: \"{out_file_name}\"")


def four_partitions():
    plot_fn = plot.plot_fit
    fig = None
    d = dict(fig=fig, plot_fn=plot_fn)
    d['fig'], ax = add_plot("results/wrn_16x4_p4_stale_weights_1.json",
                            "4p_stale_5_warmup", fig=fig, plot_fn=plot_fn)
    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_stale_weights_15_warmup.json", "4p_stale_15_warmup", **d)

    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_stale_weights_35_warmup.json", "4p_stale_35_warmup", **d)
    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_stale_weights_35_warmup_low_lr.json", "4p_stale_35_warmup_low_lr", **d)

    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_msnag_weights_35_warmup.json", "4p_msnag_35_warmup", **d)

    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_stale_weights_5_warmup_025_grad_norm.json", "4p_stale_5_warmup_025_clip", **d)
    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_msnag_weights_5_warmup_025_grad_norm.json", "4p_msnag_5_warmup_025_clip", **d)

    # TODO: add after run done, change name to json json

    gen_plot(out_dir='results', out_base_name='4p.png')


def four_partitions_grad_norm():

    plot_fn = plot.plot_grad_norm
    fig = None
    d = dict(fig=fig, plot_fn=plot_fn)
    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_stale_weights_5_warmup_025_grad_norm.json", "4p_stale_5_warmup_025_clip", **d)
    d['fig'], ax = add_plot(
        "results/wrn_16x4_p4_msnag_weights_5_warmup_025_grad_norm.json", "4p_msnag_5_warmup_025_clip", **d)
    gen_plot(out_dir='results', out_base_name='4p_grad_norm.png')


def debug_4p_with_grad_norm():
    d = dict(fig=None, plot_fn=plot.plot_grad_norm)
    # 5 warm up  for all
    d['fig'], ax = add_plot(
        "results/stale.json", "4p_stale", **d)
    d['fig'], ax = add_plot(
        "results/stale_clip.json", "4p_stale_clip", **d)
   d['fig'], ax = add_plot(
        "results/low_lr.json", "4p_low_lr", **d)
    
    gen_plot(out_dir='results', out_base_name='4p_debug_grad_norm.png')

if __name__ == "__main__":
    # current_status()
    # gap_aware()
    four_partitions()
    four_partitions_grad_norm()
    exit(0)
