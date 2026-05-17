from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("cv2")
pytest.importorskip("joblib")
pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("PIL")
pytest.importorskip("sklearn")

import pandas as pd
from PIL import Image

from image_edit_detection.cli import main
from image_edit_detection.predict import PredictionResult, predict_image, save_prediction_report
from image_edit_detection.train import train_single_experiment
from image_edit_detection.visualizations import save_ela_evidence_visualization


def _train_unit_model(tmp_path):
    rows = []
    for index in range(4):
        rows.append(
            {
                "image_path": f"original_{index}.jpg",
                "source": "public",
                "label": 0,
                "label_name": "original",
                "manipulation_type": "none",
                "dataset_name": "unit",
                "meta_width": 96,
                "meta_height": 64,
                "noise_ela_mean": 3.0 + index,
                "noise_laplacian_var": 20.0 + index,
            }
        )
        rows.append(
            {
                "image_path": f"edited_{index}.jpg",
                "source": "public",
                "label": 1,
                "label_name": "edited",
                "manipulation_type": "splicing",
                "dataset_name": "unit",
                "meta_width": 96,
                "meta_height": 64,
                "noise_ela_mean": 30.0 + index,
                "noise_laplacian_var": 200.0 + index,
            }
        )

    train_single_experiment(
        frame=pd.DataFrame(rows),
        scope="public",
        feature_group="combined",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        models_dir=tmp_path / "models",
        tune=False,
        n_jobs=1,
    )
    return tmp_path / "models" / "public_combined_random_forest.joblib"


def test_predict_image_uses_trained_model_without_label(tmp_path):
    model_path = _train_unit_model(tmp_path)
    image_path = tmp_path / "unknown.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    result = predict_image(
        image_path=image_path,
        model_path=model_path,
    )
    assert result.prediction in {"original", "edited"}
    assert set(result.probabilities) == {"original", "edited"}

    report_path = tmp_path / "reports" / "predictions" / "latest_prediction.md"
    save_prediction_report(result, report_path)
    assert "Prediction:" in report_path.read_text(encoding="utf-8")


def test_save_prediction_report_includes_visualization_link(tmp_path):
    result = PredictionResult(
        image_path=tmp_path / "unknown.jpg",
        model_path=tmp_path / "models" / "model.joblib",
        prediction="edited",
        probabilities={"original": 0.1, "edited": 0.9},
        scope="public",
        feature_group="combined",
    )
    report_path = tmp_path / "reports" / "predictions" / "latest_prediction.md"
    visualization_path = report_path.with_name("latest_prediction_evidence.png")

    save_prediction_report(result, report_path, visualization_path=visualization_path)

    report = report_path.read_text(encoding="utf-8")
    assert "ELA-псевдомаска:" in report
    assert "![ELA evidence](latest_prediction_evidence.png)" in report
    assert "ground-truth маска" in report


def test_save_ela_evidence_visualization_creates_triptych(tmp_path):
    image_path = tmp_path / "unknown.jpg"
    output_path = tmp_path / "evidence.png"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    save_ela_evidence_visualization(image_path, output_path)

    assert output_path.exists()
    with Image.open(output_path) as evidence:
        assert evidence.mode == "RGB"
        assert evidence.size == (288, 88)


def test_cli_predict_image_creates_report_and_visualization(tmp_path):
    model_path = _train_unit_model(tmp_path)
    image_path = tmp_path / "unknown.jpg"
    report_path = tmp_path / "reports" / "predictions" / "latest_prediction.md"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    exit_code = main(
        [
            "predict-image",
            "--image",
            str(image_path),
            "--model",
            str(model_path),
            "--output",
            str(report_path),
        ]
    )

    visualization_path = report_path.with_name("latest_prediction_evidence.png")
    assert exit_code == 0
    assert visualization_path.exists()
    assert "latest_prediction_evidence.png" in report_path.read_text(encoding="utf-8")


def test_cli_predict_image_no_visualization_keeps_old_report_behavior(tmp_path):
    model_path = _train_unit_model(tmp_path)
    image_path = tmp_path / "unknown.jpg"
    report_path = tmp_path / "reports" / "predictions" / "latest_prediction.md"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    exit_code = main(
        [
            "predict-image",
            "--image",
            str(image_path),
            "--model",
            str(model_path),
            "--output",
            str(report_path),
            "--no-visualization",
        ]
    )

    assert exit_code == 0
    assert not report_path.with_name("latest_prediction_evidence.png").exists()
    assert "ELA evidence" not in report_path.read_text(encoding="utf-8")
