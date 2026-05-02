"""Dataset preparation utilities for image edit detection."""

from __future__ import annotations

import hashlib
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = ("image_path", "label", "source_image", "manipulation_type", "dataset_name")
DATASET_IMAGE_FOLDERS = (
    "source",
    "generated/original",
    "generated/edited",
    "public/original",
    "public/edited",
)


@dataclass(frozen=True)
class DatasetCounts:
    """Counts produced by a dataset preparation step."""

    original: int
    edited: int


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(root: Path) -> Iterable[Path]:
    """Yield supported image files below ``root`` in deterministic order."""

    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if is_image_file(path) and not _is_mask_or_metadata_path(path):
            yield path


def ensure_dataset_dirs(dataset_root: Path) -> None:
    """Create the expected dataset directory tree."""

    for relative in (
        "external",
        "source",
        "generated/original",
        "generated/edited",
        "public/original",
        "public/edited",
    ):
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)


def trim_dataset_folders(dataset_root: Path, limit_per_folder: int = 10) -> dict[str, int]:
    """Keep at most ``limit_per_folder`` images in each dataset image folder.

    This is a convenience for course/demo repositories: the full local datasets
    remain ignored by git, while a small local working sample can be kept around
    for quick experiments.
    """

    if limit_per_folder < 1:
        raise ValueError("limit_per_folder must be at least 1")
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    keep_by_relative = _manifest_keep_sets(dataset_root, limit_per_folder)
    removed_by_folder: dict[str, int] = {}

    for relative in DATASET_IMAGE_FOLDERS:
        folder = dataset_root / relative
        if not folder.exists():
            removed_by_folder[relative] = 0
            continue

        images = list(iter_image_files(folder))
        keep_set = keep_by_relative.get(relative)
        if keep_set is None:
            keep_paths = set(images[:limit_per_folder])
        else:
            keep_paths = set(keep_set)

        removed = 0
        for image_path in images:
            if image_path not in keep_paths:
                image_path.unlink()
                removed += 1
        removed_by_folder[relative] = removed

    _rewrite_manifest_for_existing_images(dataset_root / "generated")
    _rewrite_manifest_for_existing_images(dataset_root / "public")
    return removed_by_folder


def create_demo_source_images(
    source_dir: Path,
    count: int = 12,
    size: tuple[int, int] = (640, 480),
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    """Create deterministic demo source images with simple EXIF metadata.

    The images are synthetic, but they include gradients, texture, edges, and
    camera-like metadata. They are meant for smoke tests and demonstrations
    when the user has not collected their own originals yet.
    """

    source_dir.mkdir(parents=True, exist_ok=True)
    existing = list(iter_image_files(source_dir))
    if existing and not overwrite:
        return 0

    rng = np.random.default_rng(seed)
    width, height = size

    created = 0
    for index in range(count):
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        base = np.zeros((height, width, 3), dtype=np.float32)
        base[..., 0] = 75 + 120 * xx
        base[..., 1] = 55 + 150 * yy
        base[..., 2] = 95 + 90 * (1 - xx * yy)

        noise = rng.normal(0, 10 + index % 5, size=base.shape)
        image = np.clip(base + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image, mode="RGB")

        # Add deterministic shapes so the images contain edges and local detail.
        draw_layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        for shape_idx in range(8):
            left = int(rng.integers(0, max(1, width - 120)))
            top = int(rng.integers(0, max(1, height - 120)))
            right = min(width, left + int(rng.integers(40, 180)))
            bottom = min(height, top + int(rng.integers(40, 160)))
            color = tuple(int(v) for v in rng.integers(30, 230, size=3)) + (90,)
            if shape_idx % 2 == 0:
                ImageDrawCompat.rectangle(draw_layer, (left, top, right, bottom), color)
            else:
                ImageDrawCompat.ellipse(draw_layer, (left, top, right, bottom), color)
        pil = Image.alpha_composite(pil.convert("RGBA"), draw_layer).convert("RGB")

        exif = Image.Exif()
        exif[271] = "SyntheticCam"
        exif[272] = f"CourseProject-{index % 3}"
        exif[274] = 1
        exif[305] = "Camera Firmware 1.0"
        exif[306] = f"2026:04:{1 + index:02d} 12:00:00"

        target = source_dir / f"demo_original_{index:03d}.jpg"
        pil.save(target, format="JPEG", quality=96, exif=exif)
        created += 1

    return created


def prepare_generated_dataset(
    source_dir: Path,
    generated_root: Path,
    max_images: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> DatasetCounts:
    """Create generated original/edited folders from source images."""

    original_dir = generated_root / "original"
    edited_dir = generated_root / "edited"
    original_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    source_images = list(iter_image_files(source_dir))
    if max_images is not None:
        source_images = source_images[:max_images]
    if not source_images:
        raise ValueError(f"No supported images found in {source_dir}")

    rng = np.random.default_rng(seed)
    originals = 0
    edited = 0
    manifest_rows: list[dict[str, str | int]] = []

    loaded_images: list[tuple[Path, Image.Image, bytes | None]] = []
    for source_path in source_images:
        with Image.open(source_path) as image:
            image.load()
            exif_bytes = image.info.get("exif")
            loaded_images.append((source_path, image.convert("RGB"), exif_bytes))

    for index, (source_path, image, exif_bytes) in enumerate(loaded_images):
        stem = _safe_stem(source_path, index)
        original_path = original_dir / f"{stem}_original.jpg"
        if overwrite or not original_path.exists():
            _save_jpeg(image, original_path, quality=96, exif=exif_bytes)
            originals += 1
        manifest_rows.append(
            _manifest_row(
                image_path=original_path,
                root=generated_root,
                label=0,
                source_image=source_path,
                manipulation_type="none",
                dataset_name="generated",
            )
        )

        variants = {
            "crop_resize": _edit_crop_resize(image),
            "jpeg_recompression": image.copy(),
            "blur_contrast": _edit_blur_contrast(image),
            "brightness_sharpness": _edit_brightness(image),
            "splicing": _edit_splice(image, rng),
            "metadata_removed": image.copy(),
        }

        for operation, edited_image in variants.items():
            quality = 58 if operation == "jpeg_recompression" else 88
            target_path = edited_dir / f"{stem}_{operation}.jpg"
            if overwrite or not target_path.exists():
                # Edited images are saved without EXIF on purpose: many editors
                # strip or rewrite metadata, which is a useful signal here.
                _save_jpeg(edited_image, target_path, quality=quality, exif=None)
                edited += 1
            manifest_rows.append(
                _manifest_row(
                    image_path=target_path,
                    root=generated_root,
                    label=1,
                    source_image=source_path,
                    manipulation_type=operation,
                    dataset_name="generated",
                )
            )

    _write_manifest(generated_root / "manifest.csv", manifest_rows)
    return DatasetCounts(original=originals, edited=edited)


def import_columbia_dataset(
    source_root: Path,
    public_root: Path,
    overwrite: bool = False,
) -> DatasetCounts:
    """Import Columbia-style authentic/spliced folders into project layout."""

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    original_dir = public_root / "original"
    edited_dir = public_root / "edited"
    original_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    original_sources, edited_sources = _find_public_label_dirs(source_root)
    if not original_sources or not edited_sources:
        raise ValueError(
            "Could not find authentic/spliced folders. Expected names such as "
            "4cam_auth and 4cam_splc, or Au and Sp."
        )

    original_count, original_entries = _copy_labeled_images(original_sources, original_dir, overwrite)
    edited_count, edited_entries = _copy_labeled_images(edited_sources, edited_dir, overwrite)
    _write_or_update_public_manifest(
        public_root=public_root,
        imported_originals=original_entries,
        imported_edited=edited_entries,
        dataset_name="columbia",
        edited_manipulation_type="splicing",
    )
    return DatasetCounts(original=original_count, edited=edited_count)


def import_external_dataset(
    kind: str,
    source_root: Path,
    public_root: Path,
    overwrite: bool = False,
) -> DatasetCounts:
    """Import a supported external forensic dataset into ``datasets/public``."""

    kind_normalized = kind.lower()
    if kind_normalized == "columbia":
        return import_columbia_dataset(source_root, public_root, overwrite=overwrite)

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    original_dir = public_root / "original"
    edited_dir = public_root / "edited"
    original_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    if kind_normalized == "casia":
        original_sources, edited_sources = _find_casia_label_dirs(source_root)
        manipulation_type = "tampering"
    elif kind_normalized == "comofod":
        original_sources, edited_sources = _find_comofod_files(source_root)
        manipulation_type = "copy_move"
    elif kind_normalized == "coverage":
        original_sources, edited_sources = _find_coverage_files(source_root)
        manipulation_type = "copy_move"
    elif kind_normalized == "tif-pairs":
        original_sources, edited_sources = _find_tif_pair_files(source_root)
        manipulation_type = "tampering"
    else:
        raise ValueError("kind must be one of: columbia, casia, comofod, coverage, tif-pairs")

    if not original_sources or not edited_sources:
        raise ValueError(f"Could not find original/edited images for {kind_normalized}")

    original_count, original_entries = _copy_labeled_inputs(original_sources, original_dir, overwrite)
    edited_count, edited_entries = _copy_labeled_inputs(edited_sources, edited_dir, overwrite)
    _write_or_update_public_manifest(
        public_root=public_root,
        imported_originals=original_entries,
        imported_edited=edited_entries,
        dataset_name=kind_normalized,
        edited_manipulation_type=manipulation_type,
    )
    return DatasetCounts(original=original_count, edited=edited_count)


def _edit_crop_resize(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_margin_x = max(1, int(width * 0.08))
    crop_margin_y = max(1, int(height * 0.08))
    cropped = image.crop(
        (crop_margin_x, crop_margin_y, width - crop_margin_x, height - crop_margin_y)
    )
    return cropped.resize(image.size, Image.Resampling.LANCZOS)


def _edit_blur_contrast(image: Image.Image) -> Image.Image:
    edited = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    return ImageEnhance.Contrast(edited).enhance(1.25)


def _edit_brightness(image: Image.Image) -> Image.Image:
    edited = ImageEnhance.Brightness(image).enhance(1.18)
    return ImageEnhance.Sharpness(edited).enhance(1.35)


def _edit_splice(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    patch_w = max(24, width // 5)
    patch_h = max(24, height // 5)
    src_left = int(rng.integers(0, max(1, width - patch_w)))
    src_top = int(rng.integers(0, max(1, height - patch_h)))
    dst_left = int(rng.integers(0, max(1, width - patch_w)))
    dst_top = int(rng.integers(0, max(1, height - patch_h)))

    patch = image.crop((src_left, src_top, src_left + patch_w, src_top + patch_h))
    patch = ImageOps.mirror(patch)
    patch = ImageEnhance.Color(patch).enhance(1.15)

    edited = image.copy()
    edited.paste(patch, (dst_left, dst_top))
    return edited


def _save_jpeg(
    image: Image.Image,
    target_path: Path,
    quality: int,
    exif: bytes | None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"format": "JPEG", "quality": quality, "optimize": True}
    if exif:
        save_kwargs["exif"] = exif
    image.convert("RGB").save(target_path, **save_kwargs)


def _safe_stem(path: Path, index: int) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    return f"{index:04d}_{stem}_{digest}"


def _find_public_label_dirs(source_root: Path) -> tuple[list[Path], list[Path]]:
    original_markers = {"4cam_auth", "auth", "authentic", "original", "orig", "au"}
    edited_markers = {"4cam_splc", "splc", "spliced", "tampered", "edited", "forged", "fake", "sp", "tp"}

    original_dirs: list[Path] = []
    edited_dirs: list[Path] = []

    for directory in [source_root, *sorted(p for p in source_root.rglob("*") if p.is_dir())]:
        name = directory.name.lower()
        if _is_mask_or_metadata_path(directory):
            continue
        if (
            name in original_markers
            or name.endswith("_auth")
            or name.startswith(("au-", "auth-", "authentic-", "original-"))
        ):
            original_dirs.append(directory)
        elif (
            name in edited_markers
            or name.endswith("_splc")
            or name.startswith(("sp-", "splc-", "spliced-", "tp-", "tampered-", "forged-"))
        ):
            edited_dirs.append(directory)

    return original_dirs, edited_dirs


def _find_casia_label_dirs(source_root: Path) -> tuple[list[Path], list[Path]]:
    original_dirs: list[Path] = []
    edited_dirs: list[Path] = []
    for directory in [source_root, *sorted(p for p in source_root.rglob("*") if p.is_dir())]:
        if _is_mask_or_metadata_path(directory):
            continue
        name = directory.name.lower()
        if name in {"au", "authentic", "original"}:
            original_dirs.append(directory)
        elif name in {"tp", "tampered", "edited", "forged"}:
            edited_dirs.append(directory)
    return original_dirs, edited_dirs


def _find_comofod_files(source_root: Path) -> tuple[list[Path], list[Path]]:
    originals: list[Path] = []
    edited: list[Path] = []
    for path in iter_image_files(source_root):
        stem = path.stem.lower()
        if stem.endswith("_o") or "_o_" in stem:
            originals.append(path)
        elif stem.endswith("_f") or "_f_" in stem:
            edited.append(path)
    return originals, edited


def _find_coverage_files(source_root: Path) -> tuple[list[Path], list[Path]]:
    originals: list[Path] = []
    edited: list[Path] = []
    for path in iter_image_files(source_root):
        lowered = path.as_posix().lower()
        stem = path.stem.lower()
        if any(token in lowered for token in ("/original", "/authentic", "/real")) or stem.endswith("_orig"):
            originals.append(path)
        elif any(token in lowered for token in ("/forged", "/tampered", "/edited", "/fake")) or stem.endswith("_forged"):
            edited.append(path)
    if not originals or not edited:
        return _find_comofod_files(source_root)
    return originals, edited


def _find_tif_pair_files(source_root: Path) -> tuple[list[Path], list[Path]]:
    images = list(iter_image_files(source_root))
    by_stem = {path.stem.lower(): path for path in images}
    originals: list[Path] = []
    edited: list[Path] = []

    for path in images:
        stem = path.stem.lower()
        if stem.endswith("t") and stem[:-1] in by_stem:
            edited.append(path)
        elif f"{stem}t" in by_stem:
            originals.append(path)

    return originals, edited


def _copy_labeled_images(
    source_dirs: list[Path],
    target_dir: Path,
    overwrite: bool,
) -> tuple[int, list[tuple[Path, Path]]]:
    copied = 0
    seen_targets: set[Path] = set()
    entries: list[tuple[Path, Path]] = []
    for source_dir in source_dirs:
        for source_path in iter_image_files(source_dir):
            relative_token = "_".join(source_path.relative_to(source_dir).parts)
            target_name = f"{source_dir.name}_{relative_token}"
            target_name = target_name.replace(" ", "_")
            target_path = target_dir / target_name
            if target_path in seen_targets:
                target_path = target_dir / f"{source_path.stem}_{_path_digest(source_path)}{source_path.suffix}"
            seen_targets.add(target_path)
            entries.append((source_path, target_path))
            if overwrite or not target_path.exists():
                shutil.copy2(source_path, target_path)
                copied += 1
    return copied, entries


def _copy_labeled_inputs(
    source_paths: list[Path],
    target_dir: Path,
    overwrite: bool,
) -> tuple[int, list[tuple[Path, Path]]]:
    if not source_paths:
        return 0, []
    if all(path.is_dir() for path in source_paths):
        return _copy_labeled_images(source_paths, target_dir, overwrite)

    copied = 0
    seen_targets: set[Path] = set()
    entries: list[tuple[Path, Path]] = []
    for source_path in source_paths:
        if source_path.is_dir():
            nested = list(iter_image_files(source_path))
        else:
            nested = [source_path]
        for image_path in nested:
            if not is_image_file(image_path):
                continue
            target_name = f"{image_path.stem}_{_path_digest(image_path)}{image_path.suffix}"
            target_path = target_dir / target_name.replace(" ", "_")
            if target_path in seen_targets:
                target_path = target_dir / f"{image_path.stem}_{_path_digest(image_path)}_dup{image_path.suffix}"
            seen_targets.add(target_path)
            entries.append((image_path, target_path))
            if overwrite or not target_path.exists():
                shutil.copy2(image_path, target_path)
                copied += 1
    return copied, entries


def _write_public_manifest(
    public_root: Path,
    dataset_name: str,
    edited_manipulation_type: str,
) -> None:
    rows: list[dict[str, str | int]] = []
    for label_name, label, manipulation_type in (
        ("original", 0, "none"),
        ("edited", 1, edited_manipulation_type),
    ):
        for image_path in iter_image_files(public_root / label_name):
            rows.append(
                _manifest_row(
                    image_path=image_path,
                    root=public_root,
                    label=label,
                    source_image="",
                    manipulation_type=manipulation_type,
                    dataset_name=dataset_name,
                )
            )
    _write_manifest(public_root / "manifest.csv", rows)


def _write_or_update_public_manifest(
    public_root: Path,
    imported_originals: list[tuple[Path, Path]],
    imported_edited: list[tuple[Path, Path]],
    dataset_name: str,
    edited_manipulation_type: str,
) -> None:
    """Merge newly imported rows into ``datasets/public/manifest.csv``."""

    manifest_path = public_root / "manifest.csv"
    rows_by_path = {
        str(row["image_path"]): row
        for row in _read_manifest(manifest_path)
        if (public_root / str(row["image_path"])).exists()
    }

    _add_manifest_fallback_rows(
        public_root=public_root,
        rows_by_path=rows_by_path,
        label_name="original",
        label=0,
        manipulation_type="none",
    )
    _add_manifest_fallback_rows(
        public_root=public_root,
        rows_by_path=rows_by_path,
        label_name="edited",
        label=1,
        manipulation_type="unknown",
    )

    for source_path, target_path in imported_originals:
        row = _manifest_row(
            image_path=target_path,
            root=public_root,
            label=0,
            source_image=source_path,
            manipulation_type="none",
            dataset_name=dataset_name,
        )
        rows_by_path[str(row["image_path"])] = row

    for source_path, target_path in imported_edited:
        row = _manifest_row(
            image_path=target_path,
            root=public_root,
            label=1,
            source_image=source_path,
            manipulation_type=edited_manipulation_type,
            dataset_name=dataset_name,
        )
        rows_by_path[str(row["image_path"])] = row

    rows = [rows_by_path[key] for key in sorted(rows_by_path)]
    _write_manifest(manifest_path, rows)


def _add_manifest_fallback_rows(
    public_root: Path,
    rows_by_path: dict[str, dict[str, str | int]],
    label_name: str,
    label: int,
    manipulation_type: str,
) -> None:
    for image_path in iter_image_files(public_root / label_name):
        relative = _relative_path_string(image_path, public_root)
        if relative in rows_by_path:
            continue
        rows_by_path[relative] = _manifest_row(
            image_path=image_path,
            root=public_root,
            label=label,
            source_image="",
            manipulation_type=manipulation_type,
            dataset_name="unknown_public",
        )


def _manifest_keep_sets(dataset_root: Path, limit_per_folder: int) -> dict[str, set[Path]]:
    keep_sets: dict[str, set[Path]] = {}
    for source in ("generated", "public"):
        source_root = dataset_root / source
        rows = _read_manifest(source_root / "manifest.csv")
        if not rows:
            continue
        for label_folder in ("original", "edited"):
            relative_folder = f"{source}/{label_folder}"
            selected = [
                Path(str(row["image_path"]))
                for row in rows
                if str(row["image_path"]).replace("\\", "/").startswith(f"{label_folder}/")
            ][:limit_per_folder]
            keep_sets[relative_folder] = {source_root / path for path in selected}
    return keep_sets


def _rewrite_manifest_for_existing_images(source_root: Path) -> None:
    manifest_path = source_root / "manifest.csv"
    rows = _read_manifest(manifest_path)
    if not rows:
        return

    existing_rows = []
    for row in rows:
        image_path = source_root / str(row["image_path"])
        if image_path.exists():
            existing_rows.append(row)
    _write_manifest(manifest_path, existing_rows)


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "image_path": row.get("image_path", ""),
                "label": row.get("label", ""),
                "source_image": row.get("source_image", ""),
                "manipulation_type": row.get("manipulation_type", ""),
                "dataset_name": row.get("dataset_name", ""),
            }
            for row in reader
            if row.get("image_path")
        ]


def _manifest_row(
    image_path: Path,
    root: Path,
    label: int,
    source_image: Path | str,
    manipulation_type: str,
    dataset_name: str,
) -> dict[str, str | int]:
    return {
        "image_path": _relative_path_string(image_path, root),
        "label": label,
        "source_image": str(source_image),
        "manipulation_type": manipulation_type,
        "dataset_name": dataset_name,
    }


def _write_manifest(manifest_path: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _relative_path_string(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return relative.as_posix()


def _path_digest(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


def _is_mask_or_metadata_path(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return any(
        "mask" in part
        or "groundtruth" in part
        or "ground_truth" in part
        or part in {"gt", "edgemask", "edgemasks"}
        for part in lowered
    )


class ImageDrawCompat:
    """Small wrapper to keep the Pillow drawing import local and obvious."""

    @staticmethod
    def rectangle(layer: Image.Image, box: tuple[int, int, int, int], color: tuple[int, ...]) -> None:
        from PIL import ImageDraw

        ImageDraw.Draw(layer).rectangle(box, fill=color)

    @staticmethod
    def ellipse(layer: Image.Image, box: tuple[int, int, int, int], color: tuple[int, ...]) -> None:
        from PIL import ImageDraw

        ImageDraw.Draw(layer).ellipse(box, fill=color)
