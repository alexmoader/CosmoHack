from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

AF_FEATURE_NAMES = [
    "I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith",
    "I4_minus_I5", "I4_local_contrast_3", "I4_local_contrast_7",
    "D45_local_contrast_3", "D45_local_contrast_7", "I4_local_std_7",
    "D45_local_std_7", "landcover", "dem", "t2m", "rh2m", "wind_speed",
]

BS_FEATURE_NAMES = (
    [f"pre_{b}" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")]
    + [f"post_{b}" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")]
    + [f"delta_{b}" for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")]
    + ["nbr_pre", "nbr_post", "dnbr", "rdnbr", "ndvi_pre", "ndvi_post", "dndvi", "ndmi_pre", "ndmi_post", "dndmi"]
    + ["scl_pre", "scl_post", "vv_pre", "vh_pre", "vv_post", "vh_post", "dvv", "dvh", "dem", "slope", "landcover"]
)

BS_CONTEXT_NAMES = [
    "dnbr_mean3", "dnbr_mean7", "dnbr_mean15", "dnbr_std7",
    "dnbr_contrast3", "dnbr_contrast7",
    "dndvi_mean7", "dndvi_std7", "dndmi_mean7", "dndmi_std7",
    "nbr_post_mean7", "nbr_post_std7",
    "delta_b12_mean7", "delta_b12_std7", "dvv_mean7", "dvh_mean7",
]

BS_V2_FEATURE_NAMES = list(BS_FEATURE_NAMES) + BS_CONTEXT_NAMES


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def _safe_index(a: np.ndarray, b: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (a - b) / (np.abs(a) + np.abs(b) + eps)


def af_paths(data_dir: Path, chip_id: str) -> tuple[Path, Path]:
    base = Path(data_dir) / "af"
    return base / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif", base / "aux" / f"{chip_id}_AUX.tif"


def extract_af_features(data_dir: Path, chip_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    viirs_path, aux_path = af_paths(Path(data_dir), chip_id)
    v = _read(viirs_path).astype(np.float32)
    aux = _read(aux_path).astype(np.float32)
    i1, i2, i3, i4, i5, solar, sensor, valid = v
    d45 = i4 - i5

    mean_i4_3 = uniform_filter(i4, 3, mode="nearest")
    mean_i4_7 = uniform_filter(i4, 7, mode="nearest")
    mean_d45_3 = uniform_filter(d45, 3, mode="nearest")
    mean_d45_7 = uniform_filter(d45, 7, mode="nearest")

    std_i4_7 = np.sqrt(np.maximum(uniform_filter(i4 * i4, 7, mode="nearest") - mean_i4_7**2, 0))
    std_d45_7 = np.sqrt(np.maximum(uniform_filter(d45 * d45, 7, mode="nearest") - mean_d45_7**2, 0))

    arrays = [
        i1, i2, i3, i4, i5, solar, sensor, d45,
        i4 - mean_i4_3, i4 - mean_i4_7,
        d45 - mean_d45_3, d45 - mean_d45_7,
        std_i4_7, std_d45_7,
        *list(aux),
    ]
    features = np.stack(arrays, axis=-1)
    features[~np.isfinite(features)] = -999.0
    # Invalid VIIRS pixels must never become fires
    features[valid <= 0] = -999.0
    return features.reshape(-1, features.shape[-1]), (int(i4.shape[0]), int(i4.shape[1]))


def bs_paths(data_dir: Path, chip_id: str) -> tuple[Path, Path, Path, Path, Path]:
    base = Path(data_dir) / "bs"
    return (
        base / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif",
        base / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif",
        base / "sentinel1_pre" / f"{chip_id}_Sentinel-1_pre.tif",
        base / "sentinel1_post" / f"{chip_id}_Sentinel-1_post.tif",
        base / "aux" / f"{chip_id}_AUX.tif",
    )


def extract_bs_features(data_dir: Path, chip_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    p_pre, p_post, p_s1pre, p_s1post, p_aux = bs_paths(Path(data_dir), chip_id)
    pre_all = _read(p_pre)
    post_all = _read(p_post)
    pre = pre_all[:9].astype(np.float32) / 10000.0
    post = post_all[:9].astype(np.float32) / 10000.0
    scl_pre = pre_all[9].astype(np.float32)
    scl_post = post_all[9].astype(np.float32)
    s1pre = _read(p_s1pre).astype(np.float32) / 100.0
    s1post = _read(p_s1post).astype(np.float32) / 100.0
    aux = _read(p_aux).astype(np.float32)

    # Band indices: B4=2, B8A=6, B11=7, B12=8
    nbr_pre = _safe_index(pre[6], pre[8])
    nbr_post = _safe_index(post[6], post[8])
    dnbr = nbr_pre - nbr_post
    rdnbr = np.clip(dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 0.01)), -5, 5)
    ndvi_pre = _safe_index(pre[6], pre[2])
    ndvi_post = _safe_index(post[6], post[2])
    ndmi_pre = _safe_index(pre[6], pre[7])
    ndmi_post = _safe_index(post[6], post[7])

    arrays = [
        *list(pre), *list(post), *list(post - pre),
        nbr_pre, nbr_post, dnbr, rdnbr,
        ndvi_pre, ndvi_post, ndvi_pre - ndvi_post,
        ndmi_pre, ndmi_post, ndmi_pre - ndmi_post,
        scl_pre, scl_post,
        *list(s1pre), *list(s1post), *(list(s1post - s1pre)),
        *list(aux),
    ]
    features = np.stack(arrays, axis=-1)
    features[~np.isfinite(features)] = -999.0
    return features.reshape(-1, features.shape[-1]), (int(nbr_pre.shape[0]), int(nbr_pre.shape[1]))


def extract_bs_features_v2(data_dir: Path, chip_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    """BS features plus local spatial context for the hierarchical model."""
    base, shape = extract_bs_features(data_dir, chip_id)
    h, w = shape
    cube = base.reshape(h, w, -1)

    def channel(name: str) -> np.ndarray:
        return cube[..., BS_FEATURE_NAMES.index(name)]

    def local_stats(array: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        mean = uniform_filter(array, size=size, mode="nearest")
        mean_square = uniform_filter(array * array, size=size, mode="nearest")
        std = np.sqrt(np.maximum(mean_square - mean * mean, 0))
        return mean, std

    dnbr = channel("dnbr")
    dnbr_mean3, _ = local_stats(dnbr, 3)
    dnbr_mean7, dnbr_std7 = local_stats(dnbr, 7)
    dnbr_mean15, _ = local_stats(dnbr, 15)
    dndvi_mean7, dndvi_std7 = local_stats(channel("dndvi"), 7)
    dndmi_mean7, dndmi_std7 = local_stats(channel("dndmi"), 7)
    nbr_post_mean7, nbr_post_std7 = local_stats(channel("nbr_post"), 7)
    delta_b12_mean7, delta_b12_std7 = local_stats(channel("delta_B12"), 7)
    dvv_mean7, _ = local_stats(channel("dvv"), 7)
    dvh_mean7, _ = local_stats(channel("dvh"), 7)

    context = np.stack(
        [
            dnbr_mean3, dnbr_mean7, dnbr_mean15, dnbr_std7,
            dnbr - dnbr_mean3, dnbr - dnbr_mean7,
            dndvi_mean7, dndvi_std7, dndmi_mean7, dndmi_std7,
            nbr_post_mean7, nbr_post_std7,
            delta_b12_mean7, delta_b12_std7, dvv_mean7, dvh_mean7,
        ],
        axis=-1,
    ).reshape(-1, len(BS_CONTEXT_NAMES))
    features = np.concatenate([base, context.astype(np.float32)], axis=1)
    features[~np.isfinite(features)] = -999.0
    return features, shape


def extract_bs_aux_masks(data_dir: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract SCL_post and Landcover masks directly for domain postprocessing."""
    p_pre, p_post, p_s1pre, p_s1post, p_aux = bs_paths(Path(data_dir), chip_id)
    post_all = _read(p_post)
    scl_post = post_all[9].astype(np.uint8)
    aux = _read(p_aux)
    landcover = aux[2].astype(np.uint8) if aux.shape[0] >= 3 else np.zeros_like(scl_post)
    return scl_post, landcover


def read_mask(data_dir: Path, kind: str, chip_id: str) -> np.ndarray:
    suffix = "MASK"
    path = Path(data_dir) / kind / "masks" / f"{chip_id}_{suffix}.tif"
    return _read(path)[0]
