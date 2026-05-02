from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("numpy")
pytest.importorskip("PIL")

from image_edit_detection.dataset import (
    create_demo_source_images,
    import_external_dataset,
    prepare_generated_dataset,
    trim_dataset_folders,
)


def test_prepare_generated_dataset_creates_original_and_edited_images(tmp_path):
    source_dir = tmp_path / "source"
    generated_root = tmp_path / "generated"

    created = create_demo_source_images(source_dir, count=2, size=(128, 96))
    counts = prepare_generated_dataset(source_dir, generated_root)

    assert created == 2
    assert counts.original == 2
    assert counts.edited == 12
    assert len(list((generated_root / "original").glob("*.jpg"))) == 2
    assert len(list((generated_root / "edited").glob("*.jpg"))) == 12

    manifest = (generated_root / "manifest.csv").read_text(encoding="utf-8")
    assert "image_path,label,source_image,manipulation_type,dataset_name" in manifest
    assert "crop_resize" in manifest
    assert "jpeg_recompression" in manifest
    assert "brightness_sharpness" in manifest
    assert "splicing" in manifest
    assert "metadata_removed" in manifest
    assert "generated" in manifest


def test_trim_dataset_folders_limits_images_and_rewrites_manifest(tmp_path):
    dataset_root = tmp_path / "datasets"
    source_dir = dataset_root / "source"
    generated_root = dataset_root / "generated"

    create_demo_source_images(source_dir, count=4, size=(128, 96))
    prepare_generated_dataset(source_dir, generated_root)

    removed = trim_dataset_folders(dataset_root, limit_per_folder=2)

    assert removed["source"] == 2
    assert len(list(source_dir.glob("*.jpg"))) == 2
    assert len(list((generated_root / "original").glob("*.jpg"))) == 2
    assert len(list((generated_root / "edited").glob("*.jpg"))) == 2

    manifest_lines = (generated_root / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 5


def test_import_external_columbia_creates_public_manifest(tmp_path):
    from PIL import Image

    source_root = tmp_path / "external" / "columbia"
    auth_dir = source_root / "4cam_auth"
    spliced_dir = source_root / "4cam_splc"
    public_root = tmp_path / "datasets" / "public"
    auth_dir.mkdir(parents=True)
    spliced_dir.mkdir(parents=True)

    Image.new("RGB", (64, 48), color=(120, 80, 40)).save(auth_dir / "auth_01.tif")
    Image.new("RGB", (64, 48), color=(130, 90, 50)).save(spliced_dir / "spliced_01.tif")

    counts = import_external_dataset(
        kind="columbia",
        source_root=source_root,
        public_root=public_root,
        overwrite=True,
    )

    assert counts.original == 1
    assert counts.edited == 1
    assert len(list((public_root / "original").glob("*.tif"))) == 1
    assert len(list((public_root / "edited").glob("*.tif"))) == 1

    manifest = (public_root / "manifest.csv").read_text(encoding="utf-8")
    assert "image_path,label,source_image,manipulation_type,dataset_name" in manifest
    assert "original/" in manifest
    assert "edited/" in manifest
    assert "splicing" in manifest
    assert "columbia" in manifest


def test_import_external_columbia_supports_imsplice_folder_names(tmp_path):
    from PIL import Image

    source_root = tmp_path / "external" / "ImSpliceDataset"
    auth_dir = source_root / "Au-SS-H"
    spliced_dir = source_root / "Sp-SS-H"
    public_root = tmp_path / "datasets" / "public"
    auth_dir.mkdir(parents=True)
    spliced_dir.mkdir(parents=True)

    Image.new("RGB", (64, 48), color=(100, 90, 80)).save(auth_dir / "auth_01.tif")
    Image.new("RGB", (64, 48), color=(150, 120, 90)).save(spliced_dir / "spliced_01.tif")

    counts = import_external_dataset(
        kind="columbia",
        source_root=source_root,
        public_root=public_root,
        overwrite=True,
    )

    assert counts.original == 1
    assert counts.edited == 1
    manifest = (public_root / "manifest.csv").read_text(encoding="utf-8")
    assert "original/Au-SS-H_auth_01.tif" in manifest
    assert "edited/Sp-SS-H_spliced_01.tif" in manifest
    assert "splicing" in manifest


def test_import_external_tif_pairs_uses_t_suffix_as_edited(tmp_path):
    from PIL import Image

    source_root = tmp_path / "external" / "image"
    public_root = tmp_path / "datasets" / "public"
    source_root.mkdir(parents=True)

    Image.new("RGB", (64, 48), color=(100, 90, 80)).save(source_root / "1.tif")
    Image.new("RGB", (64, 48), color=(150, 120, 90)).save(source_root / "1t.tif")

    counts = import_external_dataset(
        kind="tif-pairs",
        source_root=source_root,
        public_root=public_root,
        overwrite=True,
    )

    assert counts.original == 1
    assert counts.edited == 1
    manifest = (public_root / "manifest.csv").read_text(encoding="utf-8")
    assert "original/1_" in manifest
    assert "edited/1t_" in manifest
    assert "tampering" in manifest
    assert "tif-pairs" in manifest
