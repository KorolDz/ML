from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("cv2")
pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

from PIL import Image

from image_edit_detection.features import build_feature_table, extract_image_features, feature_columns


def test_extract_image_features_handles_image_without_exif(tmp_path):
    image_path = tmp_path / "plain.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    features = extract_image_features(image_path)

    assert features["meta_exif_present"] == 0
    assert features["meta_width"] == 96
    assert features["meta_height"] == 64
    assert "noise_laplacian_var" in features
    assert "noise_ela_mean" in features
    assert "noise_ela_hot_pixel_ratio" in features
    assert "noise_ela_block_std_mean" in features
    assert "noise_ela_block_std_max" in features
    assert "noise_residual_block_std_mean" in features
    assert "noise_residual_block_std_range" in features


def test_build_feature_table_has_expected_labels_and_feature_groups(tmp_path):
    dataset_root = tmp_path / "dataset"
    original_dir = dataset_root / "generated" / "original"
    edited_dir = dataset_root / "generated" / "edited"
    original_dir.mkdir(parents=True)
    edited_dir.mkdir(parents=True)

    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(original_dir / "original.jpg")
    Image.new("RGB", (96, 64), color=(130, 90, 50)).save(edited_dir / "edited.jpg")
    (dataset_root / "generated" / "manifest.csv").write_text(
        "\n".join(
            [
                "image_path,label,source_image,manipulation_type,dataset_name",
                "original/original.jpg,0,,none,generated",
                "edited/edited.jpg,1,original/original.jpg,blur_contrast,generated",
            ]
        ),
        encoding="utf-8",
    )

    frame = build_feature_table(dataset_root, sources=("generated",))

    assert set(frame["label_name"]) == {"original", "edited"}
    assert set(frame["label"]) == {0, 1}
    assert set(frame["manipulation_type"]) == {"none", "blur_contrast"}
    assert set(frame["dataset_name"]) == {"generated"}
    assert feature_columns(frame, "metadata")
    assert feature_columns(frame, "noise")
    assert len(feature_columns(frame, "combined")) > len(feature_columns(frame, "metadata"))
    assert "manipulation_type" not in feature_columns(frame, "combined")
    assert "dataset_name" not in feature_columns(frame, "combined")
