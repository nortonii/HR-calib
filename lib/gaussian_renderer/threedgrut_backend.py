import importlib
import importlib.util
import io
import math
import os
import shutil
import sys
import tarfile
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lib.scene import Camera, GaussianModel, LiDARSensor
from lib.utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from lib.utils.graphics_utils import camera_to_rays


_THREEDGRUT_COMMIT = "6f2176b1d8b837e556ff7c8b40bbba354c9945a4"
_THREEDGRUT_URL = f"https://codeload.github.com/nv-tlabs/3dgrut/tar.gz/{_THREEDGRUT_COMMIT}"
_OPTIX_DEV_URL = "https://codeload.github.com/NVIDIA/optix-dev/tar.gz/refs/tags/v7.5.0"
_TCNN_URL = "https://codeload.github.com/NVlabs/tiny-cuda-nn/tar.gz/refs/heads/master"
_TRACER_CACHE = {}


def _cache_root() -> Path:
    env = os.environ.get("HR_TINY_3DGRT_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "hr-tiny" / "3dgrut" / _THREEDGRUT_COMMIT


def _torch_extensions_dir() -> Path:
    return _cache_root() / "torch_extensions"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(out_path)


def _safe_extract_tarball(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            target = dest_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            src = archive.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as f:
                shutil.copyfileobj(src, f)


def _is_valid_source_tree(source_dir: Path) -> bool:
    required_files = [
        source_dir / "threedgut_tracer" / "setup_3dgut.py",
        source_dir / "threedgrt_tracer" / "tracer.py",
        source_dir / "threedgrut" / "datasets" / "protocols.py",
    ]
    for path in required_files:
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def _ensure_source_tree() -> Path:
    cache_root = _cache_root()
    source_dir = cache_root / "source"
    marker = source_dir / ".ready"
    if marker.exists() and _is_valid_source_tree(source_dir):
        return source_dir
    if marker.exists() and not _is_valid_source_tree(source_dir):
        try:
            marker.unlink()
        except OSError:
            pass

    archive_dir = cache_root / "archives"
    main_archive = archive_dir / f"3dgrut-{_THREEDGRUT_COMMIT}.tar.gz"
    optix_archive = archive_dir / "optix-dev-v7.5.0.tar.gz"
    tcnn_archive = archive_dir / "tiny-cuda-nn-master.tar.gz"

    if not main_archive.exists():
        _download_file(_THREEDGRUT_URL, main_archive)
    if not optix_archive.exists():
        _download_file(_OPTIX_DEV_URL, optix_archive)
    if not tcnn_archive.exists():
        _download_file(_TCNN_URL, tcnn_archive)

    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_tarball(main_archive, source_dir)

    optix_dir = source_dir / "threedgrt_tracer" / "dependencies" / "optix-dev"
    if not optix_dir.exists() or not any(optix_dir.iterdir()):
        optix_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_tarball(optix_archive, optix_dir)

    tcnn_dir = source_dir / "thirdparty" / "tiny-cuda-nn"
    if not tcnn_dir.exists() or not any(tcnn_dir.iterdir()):
        tcnn_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_tarball(tcnn_archive, tcnn_dir)

    marker.write_text("ok\n", encoding="utf-8")
    return source_dir


def _install_ncore_shim_if_missing() -> None:
    if importlib.util.find_spec("ncore") is not None:
        return

    from enum import Enum

    class _ShutterType(Enum):
        ROLLING_TOP_TO_BOTTOM = 0
        ROLLING_LEFT_TO_RIGHT = 1
        ROLLING_BOTTOM_TO_TOP = 2
        ROLLING_RIGHT_TO_LEFT = 3
        GLOBAL = 4

    class _PolynomialType(Enum):
        PIXELDIST_TO_ANGLE = 0
        ANGLE_TO_PIXELDIST = 1

    class _FThetaCameraModelParameters:
        PolynomialType = _PolynomialType

    ncore_mod = types.ModuleType("ncore")
    data_mod = types.ModuleType("ncore.data")
    data_mod.ShutterType = _ShutterType
    data_mod.FThetaCameraModelParameters = _FThetaCameraModelParameters
    ncore_mod.data = data_mod
    sys.modules["ncore"] = ncore_mod
    sys.modules["ncore.data"] = data_mod


def _install_threedgrut_protocols_shim(source_dir: Path) -> None:
    if "threedgrut.datasets.protocols" in sys.modules:
        return

    datasets_dir = source_dir / "threedgrut" / "datasets"
    protocols_path = datasets_dir / "protocols.py"
    if not protocols_path.exists():
        raise RuntimeError(f"Missing 3DGRUT protocols shim source at {protocols_path}")

    datasets_mod = types.ModuleType("threedgrut.datasets")
    datasets_mod.__path__ = [str(datasets_dir)]
    sys.modules["threedgrut.datasets"] = datasets_mod

    spec = importlib.util.spec_from_file_location("threedgrut.datasets.protocols", protocols_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load 3DGRUT protocols shim from {protocols_path}")

    protocols_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocols_mod)
    datasets_mod.protocols = protocols_mod
    sys.modules["threedgrut.datasets.protocols"] = protocols_mod


def _patch_threedgut_setup(source_dir: Path) -> None:
    setup_path = source_dir / "threedgut_tracer" / "setup_3dgut.py"
    if not setup_path.exists():
        raise RuntimeError(f"Missing 3DGUT setup script at {setup_path}")

    lines = setup_path.read_text(encoding="utf-8").splitlines()

    helper_line = "    capability = torch.cuda.get_device_capability()"
    if helper_line not in lines:
        insert_at = None
        for idx, line in enumerate(lines):
            if line == '        return "true" if value else "false"':
                insert_at = idx + 1
                break
        if insert_at is None:
            raise RuntimeError(f"Unable to patch TCNN half-precision helper in {setup_path}")
        lines[insert_at:insert_at] = [
            "",
            helper_line,
            "    tcnn_half_precision = int((capability[0] * 10 + capability[1]) >= 75)",
        ]

    def _patch_flag_block(block_name: str) -> None:
        start = None
        end = None
        for idx, line in enumerate(lines):
            if line == f"    {block_name} = [":
                start = idx
                break
        if start is None:
            raise RuntimeError(f"Unable to find {block_name} in {setup_path}")
        for idx in range(start + 1, len(lines)):
            if lines[idx] == "    ]":
                end = idx
                break
        if end is None:
            raise RuntimeError(f"Unable to find end of {block_name} in {setup_path}")

        block = [line for line in lines[start + 1:end] if "TCNN_HALF_PRECISION" not in line]
        insert_at = None
        for idx, line in enumerate(block):
            if '"-DTCNN_MIN_GPU_ARCH=70",' in line:
                insert_at = idx + 1
                break
        if insert_at is None:
            raise RuntimeError(f"Unable to insert TCNN half-precision define into {block_name}")
        block.insert(insert_at, '        f"-DTCNN_HALF_PRECISION={tcnn_half_precision}",')
        lines[start + 1:end] = block

    _patch_flag_block("cflags")
    _patch_flag_block("cuda_cflags")
    patched = "\n".join(lines) + "\n"
    tmp_path = setup_path.with_suffix(setup_path.suffix + ".tmp")
    tmp_path.write_text(patched, encoding="utf-8")
    tmp_path.replace(setup_path)


def activate_threedgrut_runtime() -> Path:
    source_dir = _ensure_source_tree()
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_torch_extensions_dir()))
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    _install_ncore_shim_if_missing()
    _install_threedgrut_protocols_shim(source_dir)
    _patch_threedgut_setup(source_dir)

    slang_spec = importlib.util.find_spec("slangtorch")
    if slang_spec is not None:
        slangtorch = importlib.import_module("slangtorch")
        slang_bin = Path(slangtorch.__file__).resolve().parent / "bin"
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if str(slang_bin) not in path_parts:
            os.environ["PATH"] = f"{slang_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if shutil.which("slangc") is None:
        raise RuntimeError(
            "3DGRT/3DGUT requires slangc. Install slangtorch first, then re-run tools/warmup_3dgrut.py."
        )
    return source_dir


def _render_conf(method: str):
    base = {
        "render": {
            "method": method,
            "pipeline_type": "reference",
            "backward_pipeline_type": "referenceBwd",
            "particle_kernel_degree": 4 if method == "3dgrt" else 2,
            "particle_kernel_density_clamping": True,
            "particle_kernel_min_response": 0.0113,
            "particle_kernel_min_alpha": 1.0 / 255.0,
            "particle_kernel_max_alpha": 0.99,
            "particle_radiance_sph_degree": 3,
            "primitive_type": "instances",
            "min_transmittance": 0.001 if method == "3dgrt" else 0.0001,
            "max_consecutive_bvh_update": 15,
            "enable_normals": False,
            "enable_hitcounts": True,
            "enable_kernel_timings": False,
        }
    }
    if method == "3dgut":
        base["render"]["splat"] = {
            "rect_bounding": True,
            "tight_opacity_bounding": True,
            "tile_based_culling": True,
            "n_rolling_shutter_iterations": 5,
            "ut_alpha": 1.0,
            "ut_beta": 2.0,
            "ut_kappa": 0.0,
            "ut_in_image_margin_factor": 0.1,
            "ut_require_all_sigma_points_valid": False,
            "k_buffer_size": 0,
            "global_z_order": True,
            "fine_grained_load_balancing": False,
        }
    return OmegaConf.create(base)


def _get_tracer(method: str):
    tracer = _TRACER_CACHE.get(method)
    if tracer is not None:
        return tracer
    activate_threedgrut_runtime()
    if method == "3dgrt":
        from threedgrt_tracer import Tracer
    elif method == "3dgut":
        from threedgut_tracer import Tracer
    else:
        raise ValueError(f"Unsupported 3DGRUT tracer method '{method}'")
    tracer = Tracer(_render_conf(method))
    _TRACER_CACHE[method] = tracer
    return tracer


class _BlackBackground:
    def __call__(self, ray_to_world, rays_d, rgb, opacity, train=False):
        return rgb, opacity


class _GaussianBatchAdapter:
    def __init__(self, positions, rotations, scales, density, features, active_sh_degree):
        self.positions = positions.contiguous()
        self._rotation = rotations.contiguous()
        self._scale = scales.contiguous()
        self._density = density.contiguous()
        self._features = features.contiguous()
        self.rotation = self._rotation
        self.scale = self._scale
        self.density = self._density
        self.n_active_features = int(active_sh_degree)
        self.background = _BlackBackground()

    @property
    def num_gaussians(self):
        return int(self.positions.shape[0])

    def get_rotation(self):
        return self._rotation

    def get_scale(self):
        return self._scale

    def get_density(self):
        return self._density

    def get_features(self):
        return self._features

    @staticmethod
    def rotation_activation(value):
        return value

    @staticmethod
    def scale_activation(value):
        return value

    @staticmethod
    def density_activation(value):
        return value


def _flatten_sh_features(features: torch.Tensor) -> torch.Tensor:
    features = torch.as_tensor(features)
    if features.dim() != 3 or features.shape[-1] != 3:
        raise ValueError(f"Expected SH features with shape [N, K, 3], got {tuple(features.shape)}")
    dc = features[:, 0, :]
    if features.shape[1] == 1:
        return dc.contiguous()
    rest = features[:, 1:, :].permute(0, 2, 1).reshape(features.shape[0], -1)
    return torch.cat([dc, rest], dim=1).contiguous()


def _expand_scales_for_threedgrut(scales: torch.Tensor) -> torch.Tensor:
    scales = torch.as_tensor(scales)
    if scales.shape[1] == 3:
        return scales
    if scales.shape[1] != 2:
        raise ValueError(f"Expected Gaussian scales with 2 or 3 channels, got {tuple(scales.shape)}")
    z_pad = scales.detach().max(dim=1, keepdim=True).values
    return torch.cat([scales, z_pad], dim=1)


def _prepare_gaussians(
    frame: int,
    gaussian_assets: list[GaussianModel],
    args,
    decomp=False,
    feature_mode="lidar",
    gaussian_transform_rotation=None,
    gaussian_transform_translation=None,
):
    if decomp == "background":
        gaussian_assets = gaussian_assets[:1]
    elif decomp == "object":
        gaussian_assets = gaussian_assets[1:]

    all_means3D = []
    all_density = []
    all_scales = []
    all_features = []
    obj_rot, rot_in_local = [], []
    active_sh_degree = int(gaussian_assets[0].active_sh_degree)

    for pc in gaussian_assets:
        means3D = pc.get_world_xyz(frame)
        density = pc.get_opacity
        features = pc.get_camera_features if feature_mode == "camera" else pc.get_features
        all_means3D.append(means3D)
        all_density.append(density)
        all_scales.append(_expand_scales_for_threedgrut(pc.get_scaling))
        all_features.append(_flatten_sh_features(features))
        r1, r2 = pc.get_rotation(frame)
        obj_rot.append(r1.expand(r2.shape[0], -1))
        rot_in_local.append(r2)

    means3D = torch.cat(all_means3D, dim=0)
    density = torch.cat(all_density, dim=0)
    scales = torch.cat(all_scales, dim=0)
    features = torch.cat(all_features, dim=0)

    if decomp == "background" or not args.dynamic:
        rotations = rot_in_local[0]
    elif decomp == "object":
        obj_rot = torch.cat(obj_rot, dim=0)
        rot_in_local = torch.cat(rot_in_local, dim=0)
        rotations = quaternion_raw_multiply(None, obj_rot, F.normalize(rot_in_local, dim=1))
    else:
        obj_rot = torch.cat(obj_rot[1:], dim=0)
        rots_bkgd = rot_in_local[0]
        rot_in_local = torch.cat(rot_in_local[1:], dim=0)
        rotations = quaternion_raw_multiply(None, obj_rot, F.normalize(rot_in_local, dim=1))
        rotations = torch.cat([rots_bkgd, rotations], dim=0)

    if gaussian_transform_rotation is not None and gaussian_transform_translation is not None:
        means3D = means3D @ gaussian_transform_rotation + gaussian_transform_translation
        transform_quaternion = matrix_to_quaternion(
            gaussian_transform_rotation.T.unsqueeze(0)
        ).expand(rotations.shape[0], -1)
        rotations = quaternion_raw_multiply(None, transform_quaternion, rotations)

    return _GaussianBatchAdapter(
        positions=means3D,
        rotations=F.normalize(rotations, dim=1),
        scales=scales,
        density=density,
        features=features,
        active_sh_degree=active_sh_degree,
    )


def _identity_pose(device="cuda"):
    eye = torch.eye(4, dtype=torch.float32, device=device)
    return eye.unsqueeze(0)


def _camera_intrinsics(camera: Camera):
    if getattr(camera, "K", None) is not None:
        fx = float(camera.K[0, 0])
        fy = float(camera.K[1, 1])
        cx = float(camera.K[0, 2])
        cy = float(camera.K[1, 2])
    else:
        fx = float(camera.image_width) / (2.0 * math.tan(float(camera.FoVx) * 0.5))
        fy = float(camera.image_height) / (2.0 * math.tan(float(camera.FoVy) * 0.5))
        cx = float(camera.image_width) * 0.5
        cy = float(camera.image_height) * 0.5
    return [fx, fy, cx, cy]


def _reshape_image_for_view(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    shape = tuple(int(v) for v in image.shape)
    if image.dim() == 3:
        if shape == (height, width, 3):
            return image.contiguous()
        if shape == (3, height, width):
            return image.permute(1, 2, 0).contiguous()
        if shape == (height, 3, width):
            return image.permute(0, 2, 1).contiguous()
        if shape == (width, height, 3):
            return image.permute(1, 0, 2).contiguous()
    raise ValueError(f"Unsupported image layout {shape} for expected view {(height, width, 3)}")


def _reshape_depth_for_view(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    shape = tuple(int(v) for v in depth.shape)
    if image_dim := depth.dim():
        if image_dim == 2 and shape == (height, width):
            return depth.contiguous()
        if image_dim == 3 and shape == (1, height, width):
            return depth[0].contiguous()
        if image_dim == 3 and shape == (height, width, 1):
            return depth[..., 0].contiguous()
        if image_dim == 3 and shape == (height, 1, width):
            return depth[:, 0, :].contiguous()
        if image_dim == 3 and shape == (width, height, 1):
            return depth.permute(1, 0, 2)[..., 0].contiguous()
    raise ValueError(f"Unsupported depth layout {shape} for expected view {(height, width)}")


def render_lidar_with_3dgrt(
    frame: int,
    gaussian_assets: list[GaussianModel],
    sensor: LiDARSensor | Camera | tuple,
    background: torch.Tensor,
    args,
    scaling_modifier=1.0,
    override_color=None,
    decomp=False,
    depth_only=False,
    gaussian_transform_rotation=None,
    gaussian_transform_translation=None,
    feature_mode="lidar",
):
    activate_threedgrut_runtime()
    from threedgrut.datasets.protocols import Batch

    if override_color is not None:
        raise NotImplementedError("3DGRT backend does not support override_color yet.")
    if scaling_modifier != 1.0:
        raise NotImplementedError("3DGRT backend does not support scaling_modifier overrides yet.")

    if isinstance(sensor, Camera):
        rays_o, rays_d = camera_to_rays(sensor)
    elif isinstance(sensor, LiDARSensor):
        rays_o, rays_d = sensor.get_range_rays(frame)
    elif isinstance(sensor, tuple):
        rays_o, rays_d = sensor[0], sensor[1]
    else:
        raise ValueError("sensor type not supported")

    gaussians = _prepare_gaussians(
        frame=frame,
        gaussian_assets=gaussian_assets,
        args=args,
        decomp=decomp,
        feature_mode=feature_mode,
        gaussian_transform_rotation=gaussian_transform_rotation,
        gaussian_transform_translation=gaussian_transform_translation,
    )
    if depth_only:
        gaussians._features = gaussians._features.detach()

    tracer = _get_tracer("3dgrt")
    tracer.build_acc(gaussians, rebuild=True)
    batch = Batch(
        rays_ori=rays_o.unsqueeze(0),
        rays_dir=rays_d.unsqueeze(0),
        T_to_world=_identity_pose(device=rays_o.device),
        rays_in_world_space=True,
    )
    traced = tracer.render(gaussians, batch, train=torch.is_grad_enabled(), frame_id=int(frame))

    height, width = rays_o.shape[0], rays_o.shape[1]
    rendered_attrs = _reshape_image_for_view(traced["pred_rgb"].squeeze(0), height, width)
    depth = _reshape_depth_for_view(traced["pred_dist"].squeeze(0), height, width)
    opacity = _reshape_depth_for_view(traced["pred_opacity"].squeeze(0), height, width)
    normals = _reshape_image_for_view(traced["pred_normals"].squeeze(0), height, width)
    visibility = traced.get("mog_visibility")
    if visibility is None:
        visibility = torch.zeros((gaussians.num_gaussians,), device=depth.device, dtype=depth.dtype)
    visibility = visibility.reshape(-1, 1).to(device=depth.device, dtype=depth.dtype)

    intensity = rendered_attrs[..., 0:1]
    if rendered_attrs.shape[-1] >= 3:
        rayhit_logits = rendered_attrs[..., 1:2]
        raydrop_logits = rendered_attrs[..., 2:3]
    else:
        rayhit_logits = torch.zeros_like(intensity)
        raydrop_logits = torch.zeros_like(intensity)
    if getattr(args.opt, "use_rayhit", False) and rendered_attrs.shape[-1] >= 3:
        logits = torch.cat([rayhit_logits, raydrop_logits], dim=-1)
        raydrop_prob = F.softmax(logits, dim=-1)[..., 1:2]
    else:
        raydrop_prob = torch.sigmoid(raydrop_logits)

    return {
        "rendered_attrs": rendered_attrs,
        "rgb": rendered_attrs,
        "depth": depth,
        "intensity": intensity,
        "raydrop": raydrop_prob,
        "accumulation": opacity,
        "normal": normals,
        "final_transmittance": 1.0 - opacity.clamp(0.0, 1.0),
        "means3D": gaussians.positions,
        "accum_gaussian_weight": visibility,
    }


def render_camera_with_3dgut(
    camera: Camera,
    gaussian_assets: list[GaussianModel],
    args,
    gaussian_transform_rotation=None,
    gaussian_transform_translation=None,
):
    activate_threedgrut_runtime()
    from threedgrut.datasets.protocols import Batch

    height, width = int(camera.image_height), int(camera.image_width)
    rays_o, rays_d = camera_to_rays(camera)
    rays_o = rays_o.view(height, width, 3).contiguous()
    rays_d = rays_d.view(height, width, 3).contiguous()
    gaussians = _prepare_gaussians(
        frame=int(camera.timestamp),
        gaussian_assets=gaussian_assets,
        args=args,
        feature_mode="camera",
        gaussian_transform_rotation=gaussian_transform_rotation,
        gaussian_transform_translation=gaussian_transform_translation,
    )
    tracer = _get_tracer("3dgut")
    batch = Batch(
        rays_ori=rays_o.unsqueeze(0),
        rays_dir=rays_d.unsqueeze(0),
        T_to_world=_identity_pose(device=rays_o.device),
        rays_in_world_space=True,
        intrinsics=_camera_intrinsics(camera),
    )
    traced = tracer.render(gaussians, batch, train=torch.is_grad_enabled(), frame_id=int(camera.timestamp))
    pred_rgb = _reshape_image_for_view(traced["pred_rgb"].squeeze(0), height, width)
    pred_depth = _reshape_depth_for_view(traced["pred_dist"].squeeze(0), height, width)
    visibility = traced.get("mog_visibility")
    if visibility is None:
        visibility = torch.zeros((gaussians.num_gaussians,), device=pred_rgb.device, dtype=pred_rgb.dtype)
    visibility = visibility.to(device=pred_rgb.device)

    screenspace_points = torch.zeros_like(gaussians.positions, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    positive_depth = pred_depth > 1.0e-8
    invdepth = torch.zeros_like(pred_depth)
    invdepth[positive_depth] = 1.0 / pred_depth[positive_depth]
    return {
        "rgb": pred_rgb,
        "depth": pred_depth,
        "invdepth": invdepth,
        "screenspace_points": screenspace_points,
        "radii": visibility,
        "visibility_filter": visibility > 0,
        "num_visible": int((visibility > 0).sum().item()),
    }


def warmup_threedgrut(backends=("3dgrt", "3dgut")):
    activate_threedgrut_runtime()
    warmed = []
    for backend in backends:
        tracer = _get_tracer(str(backend))
        warmed.append(type(tracer).__name__)
    return {
        "cache_root": str(_cache_root()),
        "source_dir": str(_ensure_source_tree()),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", str(_torch_extensions_dir())),
        "warmed_backends": warmed,
    }
