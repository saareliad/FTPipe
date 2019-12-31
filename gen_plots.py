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


if __name__ == "__main__":
    wrn_16x4_4p_with_grad_norm()
    wrn_16x4_4p_with_fit_plot()
    exit(0)
