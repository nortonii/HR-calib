import numpy as np


# Nominal Velodyne HDL-64E S3 vertical layout used by KITTI/KITTI-360.
# The upper block is denser near the horizon than the lower block, so a
# uniform elevation sweep is a poor approximation for row generation.
KITTI_HDL64E_VERT_DEG_TOP_TO_BOTTOM = np.concatenate(
    [
        np.linspace(2.0, -8.333, 32, dtype=np.float32),
        np.linspace(-8.833, -24.9, 32, dtype=np.float32),
    ]
)


def get_kitti_hdl64e_beam_inclinations_rad(order: str = "bottom_to_top") -> np.ndarray:
    angles = np.deg2rad(KITTI_HDL64E_VERT_DEG_TOP_TO_BOTTOM).astype(np.float32)
    if order == "top_to_bottom":
        return angles.copy()
    if order == "bottom_to_top":
        return angles[::-1].copy()
    raise ValueError(f"Unsupported beam order: {order}")


def assign_inclinations_to_rows(
    inclinations_rad: np.ndarray,
    row_angles_top_to_bottom_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inclinations_rad = np.asarray(inclinations_rad, dtype=np.float32).reshape(-1)
    row_angles = np.asarray(row_angles_top_to_bottom_rad, dtype=np.float32).reshape(-1)
    if row_angles.ndim != 1 or row_angles.size < 2:
        raise ValueError("row_angles_top_to_bottom_rad must contain at least 2 values")

    # Nearest-beam assignment matches the actual discrete laser stack better
    # than forcing points onto a linearly interpolated elevation grid.
    row_idx = np.abs(inclinations_rad[:, None] - row_angles[None, :]).argmin(axis=1)

    top_margin = 0.5 * abs(float(row_angles[0] - row_angles[1]))
    bottom_margin = 0.5 * abs(float(row_angles[-2] - row_angles[-1]))
    valid = (
        (inclinations_rad <= float(row_angles[0]) + top_margin)
        & (inclinations_rad >= float(row_angles[-1]) - bottom_margin)
    )
    return row_idx.astype(np.int32), valid
