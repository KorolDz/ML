from __future__ import annotations

pytest = __import__("pytest")
pytest.importorskip("pandas")

from image_edit_detection.report import build_course_report


def test_build_course_report_fills_metrics_table(tmp_path):
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    template = reports_dir / "course_report_template.md"
    metrics = reports_dir / "metrics.csv"
    output = reports_dir / "course_report.md"

    reports_dir.mkdir()
    figures_dir.mkdir()
    template.write_text("## Results\n\n{{METRICS_TABLE}}\n\n{{ELA_SAMPLES_SECTION}}", encoding="utf-8")
    metrics.write_text(
        "\n".join(
            [
                "scope,feature_group,status,accuracy,precision,recall,f1,roc_auc",
                "generated,combined,ok,1.0,0.9,0.8,0.85,0.95",
            ]
        ),
        encoding="utf-8",
    )

    build_course_report(template, metrics, output, figures_dir)

    report = output.read_text(encoding="utf-8")
    assert "Лучший запуск: **generated / combined**" in report
    assert "| generated | combined | 1.000 | 0.900 | 0.800 | 0.850 | 0.950 |" in report
    assert "Результат проверки изображений" in report
