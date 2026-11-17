"""The judge's own evaluation of trained Gaussians (never the tree under test's code).

    gs_eval.py --gaussians FILE.gaussians.pt [FILE ...] --out FILE.json -- <recipe arguments>

Runs with the training environment's Python from a copy of the REFERENCE tree, with the reference's CUDA extensions
(GS_BENCH_EXT). For every file: the saved raw parameters (_xyz, _features_dc, _features_rest, _opacity, _scaling,
_rotation, the active SH degree) are loaded into the reference GaussianModel; every test camera of the recipe's scene
(the --eval split: every 8th image) is rendered by the reference gaussian_renderer.render on a black background;
render and ground truth are quantized to 8 bits the way upstream's render.py saves them (torchvision save_image:
x * 255 + 0.5, clamped, truncated) and scored as upstream's metrics.py does: PSNR (utils.image_utils.psnr), SSIM
(utils.loss_utils.ssim), LPIPS with VGG (lpipsPyTorch, weights from TORCH_HOME). The means over the test views and the
number of Gaussians are written to --out."""
import argparse
import json
import os
import sys

cut = sys.argv.index('--')
p = argparse.ArgumentParser()
p.add_argument('--gaussians', nargs='+', required=True)
p.add_argument('--out', required=True)
p.add_argument('--model-path', required=True)
A = p.parse_args(sys.argv[1:cut])
TRAIN_ARGV = sys.argv[cut + 1:]

sys.path.insert(0, os.getcwd())
if os.environ.get('GS_BENCH_EXT'):
    sys.path.insert(0, os.environ['GS_BENCH_EXT'])

import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.cuda.set_device(torch.device('cuda:0'))

from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from lpipsPyTorch import lpips  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401,E402
    SEPARATE_SH = True
except ImportError:
    SEPARATE_SH = False


def quantize(img):
    return (img * 255 + 0.5).clamp(0, 255).to(torch.uint8).float() / 255


def main():
    parser = argparse.ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    OptimizationParams(parser)
    args = parser.parse_args(TRAIN_ARGV)
    args.model_path = A.model_path
    os.makedirs(A.model_path, exist_ok=True)
    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, 'default')
    scene = Scene(dataset, gaussians, shuffle=False)
    views = scene.getTestCameras()
    if not views:
        raise SystemExit('the recipe has no test cameras (--eval missing?)')
    background = torch.zeros(3, device='cuda')
    results = []
    for f in A.gaussians:
        blob = torch.load(f, map_location='cuda', weights_only=True)
        if blob['max_sh_degree'] != dataset.sh_degree or not 0 <= blob['active_sh_degree'] <= dataset.sh_degree:
            raise SystemExit(f'{f}: SH degrees {blob["active_sh_degree"]}/{blob["max_sh_degree"]}, '
                             f'recipe {dataset.sh_degree}')
        n = blob['xyz'].shape[0]
        shapes = {'xyz': (n, 3), 'features_dc': (n, 1, 3), 'features_rest': (n, (dataset.sh_degree + 1) ** 2 - 1, 3),
                  'opacity': (n, 1), 'scaling': (n, 3), 'rotation': (n, 4)}
        for k, shape in shapes.items():
            if tuple(blob[k].shape) != shape:
                raise SystemExit(f'{f}: {k} has shape {tuple(blob[k].shape)}, expected {shape}')
            if not torch.isfinite(blob[k]).all():
                raise SystemExit(f'{f}: {k} has non-finite values')
            setattr(gaussians, '_' + k, nn.Parameter(blob[k].float().contiguous().requires_grad_(False)))
        gaussians.active_sh_degree = blob['active_sh_degree']
        ps, ss, ls = [], [], []
        with torch.no_grad():
            for view in views:
                img = render(view, gaussians, pipe, background, use_trained_exp=False, separate_sh=SEPARATE_SH)['render']
                img = quantize(img.clamp(0, 1)).unsqueeze(0)
                gt = quantize(view.original_image[0:3].cuda().clamp(0, 1)).unsqueeze(0)
                ps.append(psnr(img, gt).mean().item())
                ss.append(ssim(img, gt).item())
                ls.append(lpips(img, gt, net_type='vgg').item())
        results.append({'file': os.path.basename(f), 'count': n, 'views': len(views),
                        'psnr': sum(ps) / len(ps), 'ssim': sum(ss) / len(ss), 'lpips': sum(ls) / len(ls)})
        print(json.dumps(results[-1]), flush=True)
    with open(A.out, 'w') as fh:
        json.dump({'status': 'ok', 'results': results, 'separate_sh': SEPARATE_SH}, fh, indent=1)


main()
