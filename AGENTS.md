# Repository Guidelines

## Project Structure & Module Organization

This repository contains PyTorch/Hugging Face classifiers for CERAD constructional praxis scoring. The top-level scripts are self-contained model pipelines:

- `dinov2_multihead_classifier.py`: CircleScore training and inference.
- `dinov2_cube_classifier.py`: CubeScore training and inference.
- `dinov2_pentagon_classifier.py`: overlapping-pentagons starter wrapper.
- `circle/`, `cube/`, and `pentagon/`: per-shape documentation, `train_data.csv`, `test_data.csv`, label workbooks, and examples where available.
- `cube/examples/`: visual examples used by the cube documentation.
- `papers/`: private/reference material; do not depend on it for runtime behavior.

Model weights belong in per-shape `checkpoints/` directories, which are ignored by git.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train CircleScore:

```bash
python dinov2_multihead_classifier.py --mode train --image_dir . --labels_csv ./circle/train_data.csv --save_dir ./circle/checkpoints --batch_size 8
```

Train CubeScore:

```bash
python dinov2_cube_classifier.py --mode train --image_dir . --labels_csv ./cube/train_data.csv --save_dir ./cube/checkpoints --batch_size 8
```

Train PentagonScore starter:

```bash
python dinov2_pentagon_classifier.py --mode train --image_dir . --labels_csv ./pentagon/train_data.csv --save_dir ./pentagon/checkpoints --batch_size 8
```

Run inference:

```bash
python dinov2_multihead_classifier.py --mode inference --image_dir ./test --model_path ./circle/checkpoints/final_model_v2.pt --output_csv predictions.csv
```

Use `--image_path` for one image, `--no_tta` to disable test-time augmentation, and reduce `--batch_size` to `4` on memory-limited machines.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Follow the existing script style: constants in `UPPER_SNAKE_CASE`, functions and variables in `snake_case`, and model/dataset classes in `PascalCase`. Keep shape-specific semantics in each script's `DIMENSION_CONFIG`; read the corresponding `circle/README.md` or `cube/README.md` before changing label categories.

No formatter or linter is configured. Keep imports grouped as standard library, third-party packages, then local definitions.

## Testing Guidelines

There is no configured unit test runner. Validate changes by running inference or training against the relevant held-out CSV, usually `circle/test_data.csv` or `cube/test_data.csv`, and compare predictions to labels. Keep generated outputs such as `test_predictions.csv` and `test_results.csv` out of commits.

## Commit & Pull Request Guidelines

Git history uses short, imperative, lowercase commit messages, for example `updating cube metrics` or `organizing files`. Keep commits focused on one model, dataset, or documentation change.

Pull requests should include a concise summary, affected shape (`circle`, `cube`, or shared), commands run, resulting metrics when relevant, and any label-schema changes. Include screenshots or example images only when visual scoring behavior changes.

## Security & Configuration Tips

Do not commit checkpoints, private papers, virtual environments, or OS/editor files. Treat medical scoring outputs as research aids; document model version, checkpoint path, and dataset split used for any reported result.
