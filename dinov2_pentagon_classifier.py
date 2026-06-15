"""
DINOv2 Multi-Head Image Classifier for Pentagon Drawing Analysis.

This starter reuses the cube classifier training/inference engine with an
overlapping-pentagons label schema. Confirm the scoring dimensions
in pentagon/README.md before training a production checkpoint.
"""

import argparse

import dinov2_cube_classifier as engine


PENTAGON_DIMENSION_CONFIG = {
    "drawing_present": {
        "categories": ["drawing", "no_drawing"],
        "descriptions": {
            "drawing": "The image contains a participant drawing",
            "no_drawing": "The image does not contain a participant drawing",
        },
        "na_allowed": False,
    },
    "two_pents": {
        "categories": ["two_pents", "not_two_pents"],
        "descriptions": {
            "two_pents": "Two recognizable pentagons are present",
            "not_two_pents": "Two recognizable pentagons are not present",
        },
        "na_allowed": False,
    },
    "overlapping": {
        "categories": ["overlapping", "not_overlapping"],
        "descriptions": {
            "overlapping": "The two pentagons overlap",
            "not_overlapping": "The two pentagons do not overlap",
        },
        "na_allowed": False,
    },
    "center_four": {
        "categories": ["center_four", "not_center_four"],
        "descriptions": {
            "center_four": "The overlap/crossing area creates a four-sided figure",
            "not_center_four": "The overlap/crossing area does not create a four-sided figure",
        },
        "na_allowed": False,
    },
}


def add_score_columns(results_df):
    """Add derived 0-3 training score and collapsed 0-1 final score."""
    point_two_pents = results_df["two_pents_pred"].eq("two_pents").astype(int)
    point_overlapping = results_df["overlapping_pred"].eq("overlapping").astype(int)
    point_center_four = results_df["center_four_pred"].eq("center_four").astype(int)

    results_df["training_score_pred"] = point_two_pents + point_overlapping + point_center_four
    results_df["final_score_pred"] = results_df["training_score_pred"].eq(3).astype(int)
    return results_df


def configure_engine():
    """Install pentagon dimensions into the reused training engine."""
    engine.DIMENSION_CONFIG = PENTAGON_DIMENSION_CONFIG
    engine.DIMENSIONS = list(PENTAGON_DIMENSION_CONFIG.keys())

    # The reused engine stores DIMENSION_CONFIG in defaults at import time.
    engine.MultiLabelDrawingDataset.__init__.__defaults__ = (PENTAGON_DIMENSION_CONFIG,)
    engine.DINOv2MultiHeadClassifier.__init__.__defaults__ = (
        PENTAGON_DIMENSION_CONFIG,
        "facebook/dinov2-base",
        0.5,
        True,
    )


def main():
    configure_engine()

    parser = argparse.ArgumentParser(
        description="DINOv2 Multi-Head Classifier for Pentagon Drawing Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dinov2_pentagon_classifier.py --mode train --image_dir . --labels_csv ./pentagon/train_data.csv --save_dir ./pentagon/checkpoints
  python dinov2_pentagon_classifier.py --mode inference --image_dir ./test --model_path ./pentagon/checkpoints/final_model.pt
        """,
    )

    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])

    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--labels_csv", type=str, help="Path to labels CSV")
    parser.add_argument("--save_dir", type=str, default="./pentagon/checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lp_epochs", type=int, default=30)
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--no_tta", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.image_dir or not args.labels_csv:
            parser.error("Training requires --image_dir and --labels_csv")

        engine.run_lpft_training(
            image_dir=args.image_dir,
            labels_csv=args.labels_csv,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            lp_epochs=args.lp_epochs,
            ft_epochs=args.ft_epochs,
            val_split=args.val_split,
            seed=args.seed,
        )
    else:
        if not args.model_path:
            parser.error("Inference requires --model_path")
        if not args.image_dir and not args.image_path:
            parser.error("Inference requires --image_dir or --image_path")

        input_data = args.image_dir or args.image_path
        results_df = engine.classify(
            input_data=input_data,
            model_path=args.model_path,
            use_tta=not args.no_tta,
        )
        results_df = add_score_columns(results_df)

        output_csv = args.output_csv or "pentagon_predictions.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"CSV saved to: {output_csv}")
        print("\nSample predictions:")
        print(results_df.head().to_string())


if __name__ == "__main__":
    main()
