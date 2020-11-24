import os

import matplotlib.pyplot as plt

from pipe.experiments.experiments import load_experiment
from pipe.experiments.analysis import plot

# Options are:
# plot.plot_grad_norm
# plot.plot_fit
# plot.plot_gap

STATS_SAVED_IN_ARGS = ('total_epoch_times', "train_epochs_times", "exp_total_time")


def try_to_move_from_cfg_to_fit_res(config, fit_res,
                                    stats_names=STATS_SAVED_IN_ARGS):
    for name in stats_names:
        if name in config:
            fit_res[name] = config[name]
            del config[name]


def add_plot(fn, legened, fig=None, plot_fn=plot.plot_fit, try_to_move=True):
    config, fit_res = load_experiment(fn)
    if try_to_move:
        try_to_move_from_cfg_to_fit_res(config, fit_res)

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


def gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir='results'):
    d = dict(fig=None, plot_fn=plot_fn)
    for n, c in fn_to_contour.items():
        d['fig'], ax = add_plot(n, c, **d)
    gen_plot(out_dir=out_dir, out_base_name=f"{out_base_name}.png")


def vit_b_16_c100():
    out_base_name = "ViT_B_16_norm"
    out_dir = "results/figs"
    plot_fn = plot.plot_grad_norm
    # data_dir = "results/vit/cifar100/"
    fn_to_contour = {
        "results/vit/cifar100/fast_dcgn_global_no_nesterov_meanstd05_vit_base_patch16_384_in21k_imagenet_384c384_8p_bw12_gpipe_acyclic_cifar100_384_gpipe_bs_512_se_16_seed_42.json": "global",
        # "results/vit/cifar100/fast_dcgn_local_prop_no_nesterov_meanstd05_vit_base_patch16_384_in21k_imagenet_384c384_8p_bw12_gpipe_acyclic_cifar100_384_gpipe_bs_512_se_16_seed_42.json": "local_prop",
        # "results/vit/cifar100/no_grad_norm_no_nesterov_meanstd05_vit_base_patch16_384_in21k_imagenet_384c384_8p_bw12_gpipe_acyclic_cifar100_384_gpipe_bs_512_se_16_seed_42.json": "no_clip"
    }
    # fn_to_contour = {os.path.join(
    #     data_dir, i): v for i, v in fn_to_contour.items()}

    gen_plot_from_dict(fn_to_contour, plot_fn, out_base_name, out_dir=out_dir)


if __name__ == "__main__":
    vit_b_16_c100()
