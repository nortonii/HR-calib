import os
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import dataloader
from lib.arguments import parse
from lib.gaussian_renderer import raytracing

args = parse('configs/pandaset/exp_300_center.yaml')
args = parse('configs/pandaset/static/1.yaml', args)

scene = dataloader.load_scene(args.source_dir, args, test=False)
lidar = scene.train_lidar
model_params, _ = torch.load(
    'output/lidar_rt/pandaset_center_300/scene_ps1/models/model_it_300.pth',
    map_location='cpu')
scene.training_setup(args.opt)
scene.restore(model_params, args.opt)

bg = scene.gaussians_assets[0]
# Move all gaussian params to cuda explicitly
bg._xyz     = bg._xyz.cuda()
bg._scaling = torch.nn.Parameter(bg._scaling.data.cuda())
bg._opacity = torch.nn.Parameter(bg._opacity.data.cuda())
bg._rotation = torch.nn.Parameter(bg._rotation.data.cuda())
bg._features_dc   = torch.nn.Parameter(bg._features_dc.data.cuda())
bg._features_rest = torch.nn.Parameter(bg._features_rest.data.cuda())
print(f"_scaling device: {bg._scaling.device}, _opacity device: {bg._opacity.device}")

background = torch.tensor([0, 0, 1], device='cuda').float()
os.makedirs('output_compare/scale_opacity', exist_ok=True)

def depth_colormap(depth_np, mask, dmin, dmax):
    img = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
    valid = mask & (depth_np > 0)
    t = np.clip((depth_np - dmin) / (dmax - dmin + 1e-6), 0, 1)
    cmap = plt.get_cmap('turbo')
    colored = (cmap(t)[..., :3] * 255).astype(np.uint8)
    img[valid] = colored[valid]
    return img

# (scale_factor, opacity_factor) — both applied as multiplier in activation space
combos = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0)]

test_frames = [f for f in [0, 10, 20] if f in lidar.range_image_return1]

for frame in test_frames:
    print(f"\n=== Frame {frame} ===")
    gt_depth = lidar.get_depth(frame).cuda()
    gt_mask  = lidar.get_mask(frame).cuda()
    gt_np    = gt_depth.cpu().numpy()
    mask_np  = gt_mask.cpu().numpy()
    dmin = float(gt_np[mask_np].min())
    dmax = float(gt_np[mask_np].max())

    rows, labels = [], []
    gt_img = depth_colormap(gt_np, mask_np, dmin, dmax)
    rows.append(gt_img); labels.append('GT')

    orig_scaling = bg._scaling.data.clone()  # on cuda
    orig_opacity = bg._opacity.data.clone()  # on cuda

    for (sf, of) in combos:
        # _scaling = log(scale), add log(sf) to multiply scale by sf
        bg._scaling.data = orig_scaling + float(np.log(sf))
        # _opacity = logit(alpha), clamp to ~0.9999 max
        bg._opacity.data = torch.clamp(orig_opacity + float(np.log(of)), max=10.0)

        with torch.no_grad():
            render_pkg = raytracing(frame, scene.gaussians_assets, lidar, background, args)
        pred = render_pkg['depth'].squeeze().cpu().numpy()

        pred_v = pred[mask_np]
        gt_v   = gt_np[mask_np]
        corr = float(np.corrcoef(pred_v, gt_v)[0,1]) if pred_v.std() > 0 else 0.0
        print(f"  scale×{sf:4.0f} opacity×{of:4.0f}: "
              f"pred={pred_v.mean():.2f}m  MAE={np.abs(pred_v-gt_v).mean():.2f}m  corr={corr:.3f}")

        rows.append(depth_colormap(pred, mask_np, dmin, dmax))
        labels.append(f'×{sf:.0f}')

    bg._scaling.data = orig_scaling
    bg._opacity.data = orig_opacity

    # Build image: stack rows vertically with label strip
    H, W = rows[0].shape[:2]
    lh = 25
    canvas = np.zeros((len(rows)*(H+lh), W, 3), dtype=np.uint8)
    for i, (r, lbl) in enumerate(zip(rows, labels)):
        y0 = i*(H+lh)
        canvas[y0:y0+lh] = 20
        canvas[y0+lh:y0+lh+H] = r

    fig, ax = plt.subplots(figsize=(W/100, len(rows)*(H+lh)/100), dpi=100)
    ax.imshow(canvas); ax.axis('off')
    for i, lbl in enumerate(labels):
        ax.text(8, i*(H+lh)+lh//2, lbl, color='white', fontsize=9, va='center',
                fontweight='bold')
    plt.tight_layout(pad=0)
    out = f'output_compare/scale_opacity/frame{frame:02d}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

print("\nDone.")
