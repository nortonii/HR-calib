import os
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch, numpy as np
from lib import dataloader
from lib.arguments import parse

args = parse('configs/pandaset/exp_300_center.yaml')
args = parse('configs/pandaset/static/1.yaml', args)

scene = dataloader.load_scene(args.source_dir, args, test=False)
lidar = scene.train_lidar

ckpt_path = 'output/lidar_rt/pandaset_center_300/scene_ps1/models/model_it_300.pth'
model_params, iteration = torch.load(ckpt_path, map_location='cpu')
scene.training_setup(args.opt)
scene.restore(model_params, args.opt)

bg = scene.gaussians_assets[0]
xyz_all = bg.get_local_xyz.detach().cpu().numpy()
print(f"BG Gaussians: {len(xyz_all):,}")
print(f"XYZ range: x=[{xyz_all[:,0].min():.1f},{xyz_all[:,0].max():.1f}]  "
      f"y=[{xyz_all[:,1].min():.1f},{xyz_all[:,1].max():.1f}]  "
      f"z=[{xyz_all[:,2].min():.1f},{xyz_all[:,2].max():.1f}]")

os.makedirs('output_compare/ply', exist_ok=True)

# ── colormap ──────────────────────────────────────────────────────────
def turbo(t):
    t = np.clip(t, 0, 1)
    r = np.clip(0.1357 + t*(4.5974-t*(42.3277-t*(130.5887-t*(150.5666-t*58.1375)))), 0,1)
    g = np.clip(0.0914 + t*(2.1856+t*(4.8052-t*(14.0195-t*(4.2109+t*2.7747)))), 0,1)
    b = np.clip(0.1067 + t*(2.5612+t*(0.2064-t*(6.3067+t*(1.5957-t*4.2706)))), 0,1)
    return (np.stack([r,g,b],1)*255).astype(np.uint8)

def depth_color(vals, vmin=1.4, vmax=80.0):
    return turbo((np.asarray(vals, dtype=np.float32) - vmin) / (vmax - vmin))

def save_pts(path, xyz, rgb):
    N = len(xyz)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(N):
            f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
    print(f"Saved {path}  ({N:,} pts)")

def save_rays(path, rays_o, rays_d, gt_depth_map, step=4, max_d=80.0):
    H, W, _ = rays_o.shape
    rs = np.arange(0, H, step)
    cs = np.arange(0, W, step)
    rr, cc = np.meshgrid(rs, cs, indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    o  = rays_o[rr, cc].numpy()
    d  = rays_d[rr, cc].numpy()
    gt = gt_depth_map[rr, cc]
    valid = gt > 0
    ed = np.where(valid, gt, max_d)
    endpoints = o + d * ed[:, None]
    N = len(o)
    verts  = np.vstack([o, endpoints])
    colors = np.vstack([np.tile([255,255,255], (N,1)).astype(np.uint8), depth_color(ed)])
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {2*N}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"element edge {N}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
        for i in range(2*N):
            f.write(f"{verts[i,0]:.4f} {verts[i,1]:.4f} {verts[i,2]:.4f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
        for i in range(N):
            f.write(f"{i} {i+N}\n")
    print(f"Saved {path}  ({N:,} rays, {int(valid.sum()):,} GT-valid, step={step})")

# ── 1. Scene Gaussians (subsample 500k) ──────────────────────────────
idx = np.random.choice(len(xyz_all), min(len(xyz_all), 500_000), replace=False)
xyz_sub = xyz_all[idx]
save_pts('output_compare/ply/scene_gaussians.ply', xyz_sub,
         depth_color(np.linalg.norm(xyz_sub, axis=1)))

# ── 2. Per-frame: GT lidar points + rays ─────────────────────────────
avail = sorted(lidar.range_image_return1.keys())
print(f"Frames available (first 10): {avail[:10]}")
targets = [f for f in [0, 10, 20] if f in avail]

for frame in targets:
    print(f"\n=== Frame {frame} ===")
    pts_world, _ = lidar.inverse_projection(frame)
    pts_np = pts_world.cpu().numpy()
    save_pts(f'output_compare/ply/frame{frame:02d}_gt_lidar.ply', pts_np,
             depth_color(np.linalg.norm(pts_np, axis=1)))

    r1 = lidar.range_image_return1[frame][..., 0].numpy()
    gt_map = np.where(r1 > 0, r1, 0.0)

    rays_o, rays_d = lidar.get_range_rays(frame)
    save_rays(f'output_compare/ply/frame{frame:02d}_rays.ply',
              rays_o.cpu(), rays_d.cpu(), gt_map, step=4)

print("\n=== Files ===")
for fn in sorted(os.listdir('output_compare/ply')):
    sz = os.path.getsize(f'output_compare/ply/{fn}') / 1024**2
    print(f"  {fn}  ({sz:.1f} MB)")
