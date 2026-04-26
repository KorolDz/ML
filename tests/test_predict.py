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

from image_edit_detection.predict import predict_image, save_prediction_report
from image_edit_detection.train import train_single_experiment


def test_predict_image_uses_trained_model_without_label(tmp_path):
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

    image_path = tmp_path / "unknown.jpg"
    Image.new("RGB", (96, 64), color=(120, 80, 40)).save(image_path, quality=90)

    result = predict_image(
        image_path=image_path,
        model_path=tmp_path / "models" / "public_combined_random_forest.joblib",
    )
    assert result.prediction in {"original", "edited"}
    assert set(result.probabilities) == {"original", "edited"}

    report_path = tmp_path / "reports" / "predictions" / "latest_prediction.md"
    save_prediction_report(result, report_path)
    assert "Prediction:" in report_path.read_text(encoding="utf-8")
