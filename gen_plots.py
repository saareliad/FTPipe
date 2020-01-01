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


def gen_plot(out_dir='results', out_base_name='current_status.png'):
    plt.plot()
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
        "results/wrn28x10_new/msnag_ga_28_10_c100.json": "msnag+ga",
        "results/wrn28x10_new/stale_28_10_c100.json": "stale",
        "results/wrn28x10_new/msnag_28_10_c100.json": "msnag",

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

if __name__ == "__main__":
    # wrn_16x4_4p_with_grad_norm()
    # wrn_16x4_4p_with_fit_plot()
    wrn_28x10_4p_with_grad_norm()
    wrn_28x10_4p_with_grad_norm_with_old()
    exit(0)
