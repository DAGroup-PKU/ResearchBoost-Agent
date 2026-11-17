"""The frozen Gaussian Splatting recipe: upstream train.py's defaults on the Tanks&Temples truck scene of upstream's
T&T+DB data set, with the train/test split of its evaluation (--eval: every 8th image is a test view). The judge runs
every tree with exactly these arguments through bench/gs_entry.py; the tree under test cannot change them.

Upstream defaults kept (arguments/__init__.py): 30,000 iterations, SH degree 3 (one degree more every 1,000
iterations), densification from iteration 500 to 15,000 every 100 iterations (gradient threshold 0.0002, percent
dense 0.01), opacity reset every 3,000 iterations, lambda_dssim 0.2, position lr 1.6e-4 -> 1.6e-6, the other lrs,
Adam (optimizer_type default), resolution -1 (the truck images, 979 x 546, are used at their size), images on the GPU, no
depth regularization, no antialiasing, black background."""
import os

ITERATIONS = 30_000
SCENE = os.path.join(os.environ.get('GS_BENCH_ASSETS', '/opt/researchboost/assets'), 'tandt', 'truck')


# every value of upstream's ModelParams and OptimizationParams given explicitly, so that the tree's own defaults
# (arguments/__init__.py) cannot change the recipe; boolean switches are upstream's defaults (off) except --eval
HPARAMS = [('sh_degree', 3), ('images', 'images'), ('depths', ''), ('resolution', -1), ('data_device', 'cuda'),
           ('iterations', ITERATIONS), ('position_lr_init', 0.00016), ('position_lr_final', 0.0000016),
           ('position_lr_delay_mult', 0.01), ('position_lr_max_steps', 30_000), ('feature_lr', 0.0025),
           ('opacity_lr', 0.025), ('scaling_lr', 0.005), ('rotation_lr', 0.001), ('exposure_lr_init', 0.01),
           ('exposure_lr_final', 0.001), ('exposure_lr_delay_steps', 0), ('exposure_lr_delay_mult', 0.0),
           ('percent_dense', 0.01), ('lambda_dssim', 0.2), ('densification_interval', 100),
           ('opacity_reset_interval', 3000), ('densify_from_iter', 500), ('densify_until_iter', 15_000),
           ('densify_grad_threshold', 0.0002), ('depth_l1_weight_init', 1.0), ('depth_l1_weight_final', 0.01),
           ('optimizer_type', 'default')]


def argv():
    out = ['--source_path', SCENE, '--eval']
    for k, v in HPARAMS:
        out += [f'--{k}', str(v)]
    return out
