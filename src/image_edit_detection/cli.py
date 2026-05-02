"""Command line interface for the course project pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_edit_detection.dataset import (
    create_demo_source_images,
    ensure_dataset_dirs,
    import_columbia_dataset,
    import_external_dataset,
    prepare_generated_dataset,
    trim_dataset_folders,
)
from image_edit_detection.features import build_feature_table, save_feature_table
from image_edit_detection.predict import predict_image, save_prediction_report
from image_edit_detection.report import build_course_report
from image_edit_detection.train import DEFAULT_FEATURE_GROUPS, DEFAULT_SCOPES, run_experiments
from image_edit_detection.visualizations import save_ela_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Image edit detection pipeline based on metadata and noise features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dirs", help="Create dataset folders.")
    init_parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))

    demo_parser = subparsers.add_parser("make-demo-source", help="Create demo source images.")
    demo_parser.add_argument("--source", type=Path, default=Path("datasets/source"))
    demo_parser.add_argument("--count", type=int, default=12)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--overwrite", action="store_true")

    generated_parser = subparsers.add_parser(
        "prepare-generated",
        help="Generate edited/original images from source images.",
    )
    generated_parser.add_argument("--source", type=Path, default=Path("datasets/source"))
    generated_parser.add_argument("--output", type=Path, default=Path("datasets/generated"))
    generated_parser.add_argument("--max-images", type=int, default=None)
    generated_parser.add_argument("--seed", type=int, default=42)
    generated_parser.add_argument("--overwrite", action="store_true")

    public_parser = subparsers.add_parser(
        "import-columbia",
        help="Import Columbia authentic/spliced folders into datasets/public.",
    )
    public_parser.add_argument("--source", type=Path, required=True)
    public_parser.add_argument("--output", type=Path, default=Path("datasets/public"))
    public_parser.add_argument("--overwrite", action="store_true")

    import_parser = subparsers.add_parser(
        "import-dataset",
        help="Import an external forensic dataset into datasets/public.",
    )
    import_parser.add_argument(
        "--kind",
        choices=["columbia", "casia", "comofod", "coverage", "tif-pairs"],
        required=True,
        help="External dataset layout to import.",
    )
    import_parser.add_argument("--source", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, default=Path("datasets/public"))
    import_parser.add_argument("--overwrite", action="store_true")

    features_parser = subparsers.add_parser("extract-features", help="Extract image features.")
    features_parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    features_parser.add_argument("--output", type=Path, default=Path("features/features.csv"))
    features_parser.add_argument("--sources", nargs="+", default=["generated", "public"])
    features_parser.add_argument("--strict", action="store_true")
    features_parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many extracted rows. Use 0 to disable.",
    )
    features_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract at most this many rows for a quick smoke test.",
    )

    train_parser = subparsers.add_parser("train", help="Train and evaluate Random Forest models.")
    train_parser.add_argument("--features", type=Path, default=Path("features/features.csv"))
    train_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    train_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    train_parser.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES))
    train_parser.add_argument("--feature-groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS))
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--test-size", type=float, default=0.25)
    train_parser.add_argument("--n-jobs", type=int, default=1)
    train_parser.add_argument("--no-tune", action="store_true")

    report_parser = subparsers.add_parser("build-report", help="Build filled Markdown report.")
    report_parser.add_argument("--template", type=Path, default=Path("reports/course_report_template.md"))
    report_parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.csv"))
    report_parser.add_argument("--features", type=Path, default=Path("features/features.csv"))
    report_parser.add_argument("--output", type=Path, default=Path("reports/course_report.md"))
    report_parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))

    trim_parser = subparsers.add_parser(
        "trim-datasets",
        help="Keep a small local sample in each datasets image folder.",
    )
    trim_parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    trim_parser.add_argument("--limit", type=int, default=10)

    ela_parser = subparsers.add_parser(
        "make-ela-samples",
        help="Create ELA sample visualizations for the report.",
    )
    ela_parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    ela_parser.add_argument("--output", type=Path, default=Path("reports/figures/ela_samples"))
    ela_parser.add_argument("--max-per-label", type=int, default=3)
    ela_parser.add_argument("--quality", type=int, default=90)

    predict_parser = subparsers.add_parser(
        "predict-image",
        help="Predict whether one image is original or edited.",
    )
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/public_combined_random_forest.joblib"),
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/predictions/latest_prediction.md"),
    )

    run_parser = subparsers.add_parser("run-all", help="Run generated data, features, and training.")
    run_parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    run_parser.add_argument("--source", type=Path, default=Path("datasets/source"))
    run_parser.add_argument("--features-output", type=Path, default=Path("features/features.csv"))
    run_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    run_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    run_parser.add_argument("--report-template", type=Path, default=Path("reports/course_report_template.md"))
    run_parser.add_argument("--report-output", type=Path, default=Path("reports/course_report.md"))
    run_parser.add_argument("--create-demo-source", action="store_true")
    run_parser.add_argument("--demo-count", type=int, default=12)
    run_parser.add_argument("--max-images", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--n-jobs", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--no-tune", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "init-dirs":
        ensure_dataset_dirs(args.dataset_root)
        print(f"Created dataset folders under {args.dataset_root}")
        return 0

    if args.command == "make-demo-source":
        created = create_demo_source_images(
            source_dir=args.source,
            count=args.count,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"Created {created} demo source images in {args.source}")
        return 0

    if args.command == "prepare-generated":
        counts = prepare_generated_dataset(
            source_dir=args.source,
            generated_root=args.output,
            max_images=args.max_images,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"Generated dataset: original={counts.original}, edited={counts.edited}")
        return 0

    if args.command == "import-columbia":
        counts = import_columbia_dataset(
            source_root=args.source,
            public_root=args.output,
            overwrite=args.overwrite,
        )
        print(f"Imported public dataset: original={counts.original}, edited={counts.edited}")
        return 0

    if args.command == "import-dataset":
        counts = import_external_dataset(
            kind=args.kind,
            source_root=args.source,
            public_root=args.output,
            overwrite=args.overwrite,
        )
        print(
            f"Imported {args.kind} dataset: "
            f"original={counts.original}, edited={counts.edited}"
        )
        return 0

    if args.command == "extract-features":
        frame = build_feature_table(
            dataset_root=args.dataset_root,
            sources=args.sources,
            strict=args.strict,
            progress_every=args.progress_every,
            limit=args.limit,
        )
        save_feature_table(frame, args.output)
        print(f"Saved {len(frame)} feature rows to {args.output}")
        return 0

    if args.command == "train":
        metrics = run_experiments(
            feature_csv=args.features,
            reports_dir=args.reports_dir,
            models_dir=args.models_dir,
            scopes=args.scopes,
            feature_groups=args.feature_groups,
            random_state=args.random_state,
            test_size=args.test_size,
            tune=not args.no_tune,
            n_jobs=args.n_jobs,
        )
        print(metrics.to_string(index=False))
        return 0

    if args.command == "build-report":
        build_course_report(
            template_path=args.template,
            metrics_csv=args.metrics,
            output_path=args.output,
            figures_dir=args.figures_dir,
            features_csv=args.features,
        )
        print(f"Saved report to {args.output}")
        return 0

    if args.command == "trim-datasets":
        removed_by_folder = trim_dataset_folders(
            dataset_root=args.dataset_root,
            limit_per_folder=args.limit,
        )
        for folder, removed in removed_by_folder.items():
            print(f"{folder}: removed {removed}")
        return 0

    if args.command == "make-ela-samples":
        ela_count = save_ela_samples(
            dataset_root=args.dataset_root,
            output_dir=args.output,
            max_per_label=args.max_per_label,
            quality=args.quality,
        )
        print(f"Saved {ela_count} ELA sample visualizations")
        return 0

    if args.command == "predict-image":
        result = predict_image(image_path=args.image, model_path=args.model)
        save_prediction_report(result, args.output)
        print(f"Prediction: {result.prediction}")
        print(f"Probability original: {result.probabilities.get('original', 0.0):.2f}")
        print(f"Probability edited: {result.probabilities.get('edited', 0.0):.2f}")
        print(f"Model: {result.model_path.name}")
        print(f"Saved prediction report to {args.output}")
        return 0

    if args.command == "run-all":
        ensure_dataset_dirs(args.dataset_root)
        if args.create_demo_source:
            created = create_demo_source_images(
                source_dir=args.source,
                count=args.demo_count,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            print(f"Demo source images created: {created}")

        counts = prepare_generated_dataset(
            source_dir=args.source,
            generated_root=args.dataset_root / "generated",
            max_images=args.max_images,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"Generated dataset: original={counts.original}, edited={counts.edited}")

        frame = build_feature_table(args.dataset_root, sources=("generated", "public"))
        save_feature_table(frame, args.features_output)
        print(f"Saved {len(frame)} feature rows to {args.features_output}")

        metrics = run_experiments(
            feature_csv=args.features_output,
            reports_dir=args.reports_dir,
            models_dir=args.models_dir,
            random_state=args.seed,
            tune=not args.no_tune,
            n_jobs=args.n_jobs,
        )
        print(metrics.to_string(index=False))

        ela_count = save_ela_samples(
            dataset_root=args.dataset_root,
            output_dir=args.reports_dir / "figures" / "ela_samples",
        )
        print(f"Saved {ela_count} ELA sample visualizations")

        build_course_report(
            template_path=args.report_template,
            metrics_csv=args.reports_dir / "metrics.csv",
            output_path=args.report_output,
            figures_dir=args.reports_dir / "figures",
            features_csv=args.features_output,
        )
        print(f"Saved report to {args.report_output}")
        return 0

    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
