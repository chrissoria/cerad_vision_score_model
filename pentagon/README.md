# PentagonScore

Starter multi-label classifier for an overlapping pentagons copy task. This scaffold follows the same DINOv2 ViT-B/14 plus multi-head classifier pattern as CircleScore and CubeScore.

## Status

This is a starter schema, not a finalized scoring protocol. The current assumption is a copy task involving two overlapping pentagons. Before training a production model, confirm the exact source instrument, form, scoring rubric, and redistribution limits.

## Scoring Rubric

Working English translation:

> (a) Two pentagons (figures with 5 sides) overlapping.
> (b) The crossing point creates a four-sided figure.

Operationally, a passing drawing should contain two recognizable five-sided figures that overlap, and the overlap/crossing region should form a four-sided figure. The current model decomposes this into separate heads so borderline errors can be analyzed rather than only producing a single pass/fail output.

## Training Score And Final Score

Training score range: `0-3`.

Use three intermediate checks during labeling/training:

- `+1`: two pentagons are present (`two_pents=two_pents`)
- `+1`: the two pentagons overlap (`overlapping=overlapping`)
- `+1`: the overlap/crossing region creates a four-sided figure (`center_four=center_four`)

After prediction, collapse to a final binary score range of `0-1`:

- training score `3` -> final score `1`
- training score `0`, `1`, or `2` -> final score `0`

The model heads are intermediate features for auditability and for deriving the `0-3` training score.

## Reference Stimulus

Do not commit an official test reference image unless the project has permission to redistribute it. For documentation, use a permissioned schematic or a locally generated non-official example, and keep the exact source form in a controlled/private location for labelers. If a reference image is added later, place it under `pentagon/examples/` and state its source/license here.

## Label Schema

Use `pentagon/pentagon_labels_with_dropdowns.xlsx` for annotation, then export or copy labels into `pentagon/train_data.csv` and `pentagon/test_data.csv` with these columns:

```csv
image,filename,drawing_present,two_pents,overlapping,center_four
```

| Dimension | Categories | Labeling question |
| --- | --- | --- |
| `image` | path | Absolute or relative path to the image. |
| `filename` | text | File basename for annotation convenience; ignored by training. |
| `drawing_present` | `drawing`, `no_drawing` | Is there any participant drawing? |
| `two_pents` | `two_pents`, `not_two_pents` | Are two recognizable pentagons present? |
| `overlapping` | `overlapping`, `not_overlapping` | Do the two pentagons overlap? |
| `center_four` | `center_four`, `not_center_four` | Does the overlap/crossing region create a four-sided figure? |

For blank pages, label `drawing_present=no_drawing` and set the other heads to the negative class. This keeps the starter compatible with the current dataset loader, which expects every trained dimension to have a value.

## Commands

Train:

```bash
python dinov2_pentagon_classifier.py --mode train --image_dir . --labels_csv ./pentagon/train_data.csv --save_dir ./pentagon/checkpoints --batch_size 8
```

Inference:

```bash
python dinov2_pentagon_classifier.py --mode inference --image_dir ./test --model_path ./pentagon/checkpoints/final_model.pt --output_csv pentagon_predictions.csv
```

## Startup Checklist

- [ ] Confirm the source instrument and item name; current assumption is an overlapping pentagons copy task.
- [ ] Confirm the exact pass/fail scoring rule from the form or study protocol.
- [ ] Finalize the label dimensions and allowed categories.
- [ ] Decide how to label blank pages and non-pentagon drawings.
- [ ] Create a written labeling guide with positive, negative, and borderline examples.
- [ ] Collect image files and update CSV paths relative to `--image_dir` or as absolute paths.
- [ ] Split labels into `pentagon/train_data.csv` and `pentagon/test_data.csv`.
- [ ] Label at least a small pilot set with two reviewers and reconcile disagreements.
- [ ] Train a smoke-test model with small epochs, for example `--lp_epochs 1 --ft_epochs 1`.
- [ ] Run inference on the held-out test set and save metrics in this README.
- [ ] Add representative example images under `pentagon/examples/` after de-identification.

## Notes

The current script reuses the cube training engine to avoid duplicating the full LP-FT pipeline. If the pentagon schema stabilizes, the next cleanup step is to extract shared classifier code into a reusable module.
