"""Forensic visualizations for report artifacts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from image_edit_detection.dataset import iter_image_files


def save_ela_samples(
    dataset_root: Path,
    output_dir: Path,
    sources: tuple[str, ...] = ("generated", "public"),
    max_per_label: int = 3,
    quality: int = 90,
) -> int:
    """Save a small set of ELA sample images for original and edited classes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for source in sources:
        for label_name in ("original", "edited"):
            label_dir = dataset_root / source / label_name
            if not label_dir.exists():
                continue
            for index, image_path in enumerate(iter_image_files(label_dir)):
                if index >= max_per_label:
                    break
                output_path = output_dir / f"{source}_{label_name}_{index:02d}_ela.png"
                save_ela_visualization(image_path, output_path, quality=quality)
                saved += 1
    return saved


def save_ela_visualization(image_path: Path, output_path: Path, quality: int = 90) -> None:
    """Create a side-by-side original/ELA image for visual inspection."""

    with Image.open(image_path) as image:
        original = image.convert("RGB")
        original.thumbnail((360, 360), Image.Resampling.LANCZOS)
        ela = _ela_image(original, quality=quality)

    canvas = Image.new("RGB", (original.width * 2, original.height), color=(255, 255, 255))
    canvas.paste(original, (0, 0))
    canvas.paste(ela.resize(original.size, Image.Resampling.LANCZOS), (original.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_ela_evidence_visualization(
    image_path: Path,
    output_path: Path,
    quality: int = 90,
    threshold_percentile: float = 95.0,
) -> None:
    """Create original/ELA/overlay evidence image for one prediction."""

    if not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100")

    with Image.open(image_path) as image:
        original = image.convert("RGB")
        original.thumbnail((360, 360), Image.Resampling.LANCZOS)
        diff = _ela_difference(original, quality=quality)

    ela = _ela_image_from_difference(diff)
    ela_gray = np.mean(diff, axis=2)
    threshold = float(np.percentile(ela_gray, threshold_percentile))
    if threshold <= 0:
        mask = ela_gray > 0
    else:
        mask = ela_gray >= threshold

    overlay = _overlay_mask(original, mask)
    canvas = _labeled_triptych(
        panels=(original, ela, overlay),
        labels=("Original", "ELA map", "ELA pseudomask"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _ela_image(image: Image.Image, quality: int = 90) -> Image.Image:
    return _ela_image_from_difference(_ela_difference(image, quality=quality))


def _ela_difference(image: Image.Image, quality: int = 90) -> np.ndarray:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        return np.abs(
            np.asarray(image, dtype=np.int16)
            - np.asarray(recompressed.convert("RGB"), dtype=np.int16)
        ).astype(np.uint8)


def _ela_image_from_difference(diff: np.ndarray) -> Image.Image:
    diff_image = Image.fromarray(diff, mode="RGB")
    return ImageOps.autocontrast(diff_image)


def _overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.45
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * red
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _labeled_triptych(panels: tuple[Image.Image, ...], labels: tuple[str, ...]) -> Image.Image:
    header_height = 24
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + header_height
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x_offset = 0
    for panel, label in zip(panels, labels):
        draw.text((x_offset + 8, 6), label, fill=(20, 20, 20), font=font)
        canvas.paste(panel, (x_offset, header_height))
        x_offset += panel.width
    return canvas
