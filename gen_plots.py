from misc import plot
from experiments import load_experiment
import matplotlib.pyplot as plt
import os
import numpy as np
# plot.plot_grad_norm
# plot.plot_fit


def add_plot(fn, legened, fig=None, plot_fn=plot.plot_fit):
    config, fit_res = load_experiment(fn)
    loss_per_batch = "loss_per_batch" in config['statistics']
    fig, ax = plot_fn(fit_res, fig=fig, log_loss=False,
                      legend=legened, loss_per_batch=loss_per_batch)
    return fig, ax


def gen_plot(out_dir='results', out_base_name='current_status.png'):
    plt.plot()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file_name = os.path.join(out_dir, out_base_name)
    plt.savefig(out_file_name)
    print(f"-I- Generated: \"{out_file_name}\"")


def wrn_16x4_4p_with_grad_norm():
    d = dict(fig=None, plot_fn=plot.plot_grad_norm)
    # 5 warm up for all
    fn_to_contour = {
        "results/stale.json": "4p_stale",
        "results/stale_clip.json": "4p_stale_clip",
        "results/stale_low_lr.json": "4p_low_lr",
        "results/msnag.json": "4p_msnag",
        "results/msnag_ga.json": "4p_msnag+ga",
    }

    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)

    gen_plot(out_dir='results', out_base_name='4p_wrn16x4_grad_norm.png')


def wrn_16x4_4p_with_fit_plot():

    d = dict(fig=None, plot_fn=plot.plot_fit)
    # 5 warm up  for all
    fn_to_contour = {
        "results/stale.json": "4p_stale",
        "results/stale_clip.json": "4p_stale_clip",
        "results/stale_low_lr.json": "4p_low_lr",
        "results/msnag.json": "4p_msnag",
        "results/msnag_ga.json": "4p_msnag+ga",
    }

    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)

    gen_plot(out_dir='results', out_base_name='4p_wrn16x4_fit_plot.png')


def wrn_28x10_4p_with_grad_norm():
    d = dict(fig=None, plot_fn=plot.plot_grad_norm)
    # 5 warm up for all
    fn_to_contour = {
        "results/wrn28x10_new/msnag_ga_28_10_c100_s2.json": "msnag+ga",
        "results/wrn28x10_new/stale_28_10_c100_s2.json": "stale",
        "results/wrn28x10_new/msnag_28_10_c100_s2.json": "msnag",
    }

    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)

    gen_plot(out_dir='results', out_base_name='4p_wrn28x4_grad_norm.png')


def wrn_28x10_4p_with_grad_norm_with_old():
    d = dict(fig=None, plot_fn=plot.plot_grad_norm)
    # 5 warm up for all
    fn_to_contour = {
        "results/wrn28x10_new/msnag_ga_28_10_c100.json": "msnag+ga",
        "results/wrn28x10_new/stale_28_10_c100.json": "stale",
        "results/wrn28x10_new/msnag_28_10_c100.json": "msnag",
        "results/wrn28x10/msnag_ga_28_10_c100.json": "old_msnag+ga",
        "results/wrn28x10/stale_28_10_c100.json": "old_stale",
        "results/wrn28x10/msnag_28_10_c100.json": "old_msnag",

    }

    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)

    gen_plot(out_dir='results', out_base_name='4p_wrn28x4_grad_norm_with_old.png')


def gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir='results'):
    d = dict(fig=None, plot_fn=plot_fn)
    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)
    gen_plot(out_dir=out_dir, out_base_name=f"{out_base_name}.png")


def wrn16x4_c100_p2():
    out_base_name = "wrn16x4_cifar100_p2"
    out_dir = "results/figs"
    plot_fn = plot.plot_grad_norm

    datadir = "results"
    fn_to_contour = {
        f"ninja4_wrn_16x4_c100_p2_cifar100_stale_seed_42.json": "stale",
        f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_seed_42.json": "nag",
        f"ninja4_wrn_16x4_c100_p2_cifar100_stale_ws_seed_42.json": "ws",
        f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_ws_seed_42.json": "nag+ws",
        f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_ws_ga_seed_42.json": "ws+nag+ga",
        # f"ninja4_wrn_16x4_p2_cifar10_msnag_seed_42.json": "",
    }

    fn_to_contour = {os.path.join(
        datadir, i): v for i, v in fn_to_contour.items()}

    gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir=out_dir)


def wrn16x4_c100_p2_gap():
    out_base_name = "wrn16x4_cifar100_p2_Gap"
    out_dir = "results/figs"
    plot_fn = plot.plot_gap
    datadir = "results"

    fn_to_contour = {
        # f"ninja4_wrn_16x4_c100_p2_cifar100_stale_seed_42.json": "stale",
        # f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_seed_42.json": "nag",
        f"ninja4_wrn_16x4_c100_p2_cifar100_stale_ws_seed_42.json": "ws",
        f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_ws_seed_42.json": "nag+ws",
        f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_ws_ga_seed_42.json": "ws+nag+ga",
    }
    fn_to_contour = {os.path.join(
        datadir, i): v for i, v in fn_to_contour.items()}

    gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir=out_dir)


def debug_gap_wrn16x4_c100():

    out_base_name = "debug_gap_wrn16x4_c100"
    out_dir = "results/figs"
    # plot_fn = plot.plot_gap
    plot_fn = plot.plot_grad_norm
    datadir = "results/4partitions/wrn16x4_cifar100"

    fn_to_contour = {
        # f"ninja4_wrn_16x4_c100_p2_cifar100_stale_seed_42.json": "stale",
        # f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_seed_42.json": "nag",
        f"agg_wrn_16x4_c100_p4_cifar100_stale_ws_bs_128_se_4_seed_1322019.json": "ws",
        f"exp_wrn_16x4_c100_p4_cifar100_msnag_ws_bs_128_se_1_seed_42.json": "nag+ws",
        # f"ninja4_wrn_16x4_c100_p2_cifar100_msnag_ws_ga_seed_42.json": "ws+nag+ga",
    }
    fn_to_contour = {os.path.join(
        datadir, i): v for i, v in fn_to_contour.items()}

    gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir=out_dir)



if __name__ == "__main__":
    # wrn_16x4_4p_with_grad_norm()
    # wrn_16x4_4p_with_fit_plot()
    # wrn_28x10_4p_with_grad_norm()
    # wrn_28x10_4p_with_grad_norm_with_old()
    # wrn16x4_c100_p2()
    # wrn16x4_c100_p2_gap()
    debug_gap_wrn16x4_c100()
    exit(0)
