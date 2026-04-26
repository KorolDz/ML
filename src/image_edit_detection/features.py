"""Feature extraction for metadata and image-noise edit detection signals."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image

from image_edit_detection.dataset import is_image_file


LABELS = {"original": 0, "edited": 1}
METADATA_PREFIX = "meta_"
NOISE_PREFIX = "noise_"
INDEX_COLUMNS = {
    "image_path",
    "source",
    "label",
    "label_name",
    "source_image",
    "manipulation_type",
    "dataset_name",
}


def extract_image_features(path: Path) -> dict[str, float | int | str]:
    """Extract one numeric feature row from an image file."""

    with Image.open(path) as image:
        image.load()
        features: dict[str, float | int | str] = {
            "image_path": str(path),
        }
        features.update(_metadata_features(path, image))
        features.update(_noise_features(image))
    return _clean_feature_dict(features)


def build_feature_table(
    dataset_root: Path,
    sources: Iterable[str] = ("generated", "public"),
    strict: bool = False,
) -> pd.DataFrame:
    """Build a feature table from the expected dataset directory layout."""

    rows: list[dict[str, float | int | str]] = []
    errors: list[str] = []

    for source_name in sources:
        source_root = dataset_root / source_name
        manifest = _load_manifest(source_root)
        for label_name, label in LABELS.items():
            label_dir = source_root / label_name
            if not label_dir.exists():
                continue
            for path in _iter_images(label_dir):
                try:
                    manifest_key = _manifest_key(path, source_root)
                    if manifest and manifest_key not in manifest:
                        continue
                    row = extract_image_features(path)
                    row["source"] = source_name
                    row["label_name"] = label_name
                    row["label"] = label
                    manifest_row = manifest.get(manifest_key, {})
                    row["source_image"] = str(manifest_row.get("source_image", ""))
                    row["manipulation_type"] = str(
                        manifest_row.get(
                            "manipulation_type",
                            _default_manipulation_type(source_name, label_name),
                        )
                    )
                    row["dataset_name"] = str(manifest_row.get("dataset_name", source_name))
                    rows.append(row)
                except Exception as exc:  # pragma: no cover - exercised by real corrupt files
                    message = f"{path}: {exc}"
                    if strict:
                        raise RuntimeError(message) from exc
                    errors.append(message)

    if not rows:
        raise ValueError(f"No image features extracted from {dataset_root}")

    frame = pd.DataFrame(rows)
    numeric_columns = [column for column in frame.columns if column not in INDEX_COLUMNS]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    if errors:
        frame.attrs["errors"] = errors
    return frame


def save_feature_table(frame: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)


def feature_columns(frame: pd.DataFrame, group: str) -> list[str]:
    """Return model input columns for a feature group."""

    if group == "metadata":
        prefixes = (METADATA_PREFIX,)
    elif group == "noise":
        prefixes = (NOISE_PREFIX,)
    elif group == "combined":
        prefixes = (METADATA_PREFIX, NOISE_PREFIX)
    else:
        raise ValueError("group must be one of: metadata, noise, combined")

    columns = [
        column
        for column in frame.columns
        if column not in INDEX_COLUMNS and column.startswith(prefixes)
    ]
    if not columns:
        raise ValueError(f"No columns found for feature group {group!r}")
    return columns


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if is_image_file(path):
            yield path


def _load_manifest(source_root: Path) -> dict[str, dict[str, object]]:
    manifest_path = source_root / "manifest.csv"
    if not manifest_path.exists():
        return {}
    frame = pd.read_csv(manifest_path).fillna("")
    if "image_path" not in frame.columns:
        return {}
    return {
        _normalize_manifest_path(str(row["image_path"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def _manifest_key(path: Path, source_root: Path) -> str:
    try:
        relative = path.relative_to(source_root)
        return _normalize_manifest_path(relative.as_posix())
    except ValueError:
        return _normalize_manifest_path(str(path))


def _normalize_manifest_path(value: str) -> str:
    return value.replace("\\", "/")


def _default_manipulation_type(source_name: str, label_name: str) -> str:
    if label_name == "original":
        return "none"
    if source_name == "public":
        return "splicing"
    return "unknown"


def _metadata_features(path: Path, image: Image.Image) -> dict[str, float | int]:
    width, height = image.size
    pixel_count = max(1, width * height)
    file_size = path.stat().st_size
    image_format = (image.format or path.suffix.lstrip(".")).upper()
    bands = image.getbands()
    exif = image.getexif()
    exif_by_name = {
        str(ExifTags.TAGS.get(tag_id, tag_id)): _stringify_exif_value(value)
        for tag_id, value in exif.items()
    }

    software = exif_by_name.get("Software", "")
    make = exif_by_name.get("Make", "")
    model = exif_by_name.get("Model", "")
    date_time = exif_by_name.get("DateTimeOriginal") or exif_by_name.get("DateTime", "")
    orientation = exif_by_name.get("Orientation", "")
    editor_keywords = ("adobe", "photoshop", "lightroom", "gimp", "paint", "canva", "snapseed")

    core_presence = [
        bool(make),
        bool(model),
        bool(date_time),
        bool(orientation),
    ]

    return {
        "meta_width": width,
        "meta_height": height,
        "meta_megapixels": pixel_count / 1_000_000,
        "meta_aspect_ratio": width / max(1, height),
        "meta_file_size_bytes": file_size,
        "meta_bytes_per_pixel": file_size / pixel_count,
        "meta_channels": len(bands),
        "meta_mode_rgb": int(image.mode == "RGB"),
        "meta_mode_l": int(image.mode == "L"),
        "meta_format_jpeg": int(image_format in {"JPEG", "JPG"}),
        "meta_format_png": int(image_format == "PNG"),
        "meta_format_tiff": int(image_format in {"TIFF", "TIF"}),
        "meta_format_bmp": int(image_format == "BMP"),
        "meta_exif_present": int(bool(exif)),
        "meta_exif_tag_count": len(exif),
        "meta_exif_has_software": int(bool(software)),
        "meta_exif_has_make": int(bool(make)),
        "meta_exif_has_model": int(bool(model)),
        "meta_exif_has_datetime": int(bool(date_time)),
        "meta_exif_has_orientation": int(bool(orientation)),
        "meta_exif_software_len": len(software),
        "meta_exif_make_len": len(make),
        "meta_exif_model_len": len(model),
        "meta_editor_software_keyword": int(any(token in software.lower() for token in editor_keywords)),
        "meta_missing_core_ratio": 1 - (sum(core_presence) / len(core_presence)),
    }


def _noise_features(image: Image.Image) -> dict[str, float | int]:
    pil_rgb = _resize_for_features(image.convert("RGB"))
    rgb = np.asarray(pil_rgb, dtype=np.float32)
    rgb_u8 = rgb.astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray64 = gray.astype(np.float64)

    residual = gray - cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(gray64, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray64, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray64, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    ela = _ela_difference(pil_rgb, quality=90)
    ela_gray = np.mean(ela, axis=2)

    block_score, block_boundary, block_non_boundary = _jpeg_block_artifact_score(gray)
    ela_block_std = _block_std_values(ela_gray)
    residual_block_std = _block_std_values(np.abs(residual))
    high_frequency_ratio = _high_frequency_energy_ratio(gray)
    entropy = _entropy(gray)

    features = {
        "noise_gray_mean": float(np.mean(gray)),
        "noise_gray_std": float(np.std(gray)),
        "noise_gray_entropy": entropy,
        "noise_laplacian_var": float(np.var(laplacian)),
        "noise_sobel_mean": float(np.mean(sobel_magnitude)),
        "noise_sobel_std": float(np.std(sobel_magnitude)),
        "noise_residual_mean": float(np.mean(residual)),
        "noise_residual_std": float(np.std(residual)),
        "noise_residual_abs_mean": float(np.mean(np.abs(residual))),
        "noise_residual_skew": _skew(residual),
        "noise_residual_kurtosis": _kurtosis(residual),
        "noise_ela_mean": float(np.mean(ela)),
        "noise_ela_std": float(np.std(ela)),
        "noise_ela_max": float(np.max(ela)),
        "noise_ela_p95": float(np.percentile(ela, 95)),
        "noise_ela_hot_pixel_ratio": _hot_pixel_ratio(ela_gray),
        "noise_ela_block_std_mean": float(np.mean(ela_block_std)) if ela_block_std.size else 0.0,
        "noise_ela_block_std_max": float(np.max(ela_block_std)) if ela_block_std.size else 0.0,
        "noise_jpeg_block_score": block_score,
        "noise_jpeg_boundary_diff": block_boundary,
        "noise_jpeg_non_boundary_diff": block_non_boundary,
        "noise_residual_block_std_mean": (
            float(np.mean(residual_block_std)) if residual_block_std.size else 0.0
        ),
        "noise_residual_block_std_range": (
            float(np.max(residual_block_std) - np.min(residual_block_std))
            if residual_block_std.size
            else 0.0
        ),
        "noise_high_frequency_ratio": high_frequency_ratio,
        "noise_saturation_mean": float(np.mean(hsv[..., 1])),
        "noise_saturation_std": float(np.std(hsv[..., 1])),
    }

    for channel_index, channel_name in enumerate(("r", "g", "b")):
        channel = rgb[..., channel_index]
        features[f"noise_{channel_name}_mean"] = float(np.mean(channel))
        features[f"noise_{channel_name}_std"] = float(np.std(channel))

    return features


def _resize_for_features(image: Image.Image, max_side: int = 512) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _ela_difference(image: Image.Image, quality: int) -> np.ndarray:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed_rgb = recompressed.convert("RGB")
        original = np.asarray(image, dtype=np.float32)
        compressed = np.asarray(recompressed_rgb, dtype=np.float32)
    return np.abs(original - compressed)


def _jpeg_block_artifact_score(gray: np.ndarray) -> tuple[float, float, float]:
    height, width = gray.shape
    diffs: list[np.ndarray] = []
    boundary_values: list[np.ndarray] = []
    non_boundary_values: list[np.ndarray] = []

    if width > 8:
        diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
        boundary_cols = (np.arange(1, width) % 8) == 0
        boundary_values.append(diff_x[:, boundary_cols])
        non_boundary_values.append(diff_x[:, ~boundary_cols])
        diffs.append(diff_x)

    if height > 8:
        diff_y = np.abs(gray[1:, :] - gray[:-1, :])
        boundary_rows = (np.arange(1, height) % 8) == 0
        boundary_values.append(diff_y[boundary_rows, :])
        non_boundary_values.append(diff_y[~boundary_rows, :])
        diffs.append(diff_y)

    if not diffs:
        return 0.0, 0.0, 0.0

    boundary = _mean_of_arrays(boundary_values)
    non_boundary = _mean_of_arrays(non_boundary_values)
    score = boundary / (non_boundary + 1e-6)
    return float(score), float(boundary), float(non_boundary)


def _block_std_values(values: np.ndarray, block_size: int = 16) -> np.ndarray:
    height, width = values.shape[:2]
    block_stds: list[float] = []
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            block = values[top : top + block_size, left : left + block_size]
            if block.size:
                block_stds.append(float(np.std(block)))
    return np.asarray(block_stds, dtype=np.float32)


def _hot_pixel_ratio(values: np.ndarray) -> float:
    threshold = float(np.percentile(values, 95))
    if threshold <= 0:
        return 0.0
    return float(np.mean(values >= threshold))


def _high_frequency_energy_ratio(gray: np.ndarray) -> float:
    centered = gray - np.mean(gray)
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.abs(spectrum) ** 2
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    high_freq_mask = radius > 0.35 * max(height, width)
    total = float(np.sum(magnitude)) + 1e-6
    return float(np.sum(magnitude[high_freq_mask]) / total)


def _entropy(gray: np.ndarray) -> float:
    histogram, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    probabilities = histogram[histogram > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _skew(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values)) + 1e-6
    return float(np.mean(((values - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values)) + 1e-6
    return float(np.mean(((values - mean) / std) ** 4))


def _mean_of_arrays(arrays: list[np.ndarray]) -> float:
    non_empty = [array for array in arrays if array.size]
    if not non_empty:
        return 0.0
    return float(np.mean(np.concatenate([array.ravel() for array in non_empty])))


def _stringify_exif_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _clean_feature_dict(features: dict[str, float | int | str]) -> dict[str, float | int | str]:
    cleaned: dict[str, float | int | str] = {}
    for key, value in features.items():
        if isinstance(value, str):
            cleaned[key] = value
        elif isinstance(value, (int, np.integer)):
            cleaned[key] = int(value)
        else:
            number = float(value)
            cleaned[key] = number if math.isfinite(number) else 0.0
    return cleaned
