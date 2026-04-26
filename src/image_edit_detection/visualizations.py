"""Forensic visualizations for report artifacts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

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


def _ela_image(image: Image.Image, quality: int = 90) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        diff = np.abs(
            np.asarray(image, dtype=np.int16)
            - np.asarray(recompressed.convert("RGB"), dtype=np.int16)
        ).astype(np.uint8)

    diff_image = Image.fromarray(diff, mode="RGB")
    return ImageOps.autocontrast(diff_image)

