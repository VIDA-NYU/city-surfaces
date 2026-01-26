# CitySurfaces User Manual

CitySurfaces combines active learning and semantic segmentation to classify sidewalk paving materials from street-level images. It uses a hierarchical multi-scale attention model with object-contextual representations and provides pretrained weights for inference.

## Install

Use Python 3.12+.

```bash
python -m pip install -U pip
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

## Quick Start

Evaluate a folder of images:

```bash
python -m citysurfaces.val --dataset citysurfaces --eval folder --eval_folder /path/to/images --result_dir results
```

## Import Check

```bash
python scripts/import_check.py
```

## Basic Eval Check

```bash
python scripts/basic_eval_check.py
```

## Key Options

- `--eval`: `val`, `folder`, or `test`.
- `--eval_folder`: required when `--eval folder`.
- `--result_dir`: output directory for dumps and logs.
- `--dump_assets`: dump predictions/attentions.
- `--dump_all_images`: dump all validation images.
- `--dump_topn`: dump worst cases by class.
- `--bs_val`: validation batch size (default 1).
- `--amp`: enable autocast during eval.
- `--seed`: set random seed.
- `--deterministic`: make cuDNN deterministic.

## Environment Variables

- `ALLOW_BLACK_FALLBACK=1` to replace unreadable images with black placeholders.
- `FAILED_READS_REPORT=/path/failed_reads.jsonl` to record unreadable image paths.

## Outputs

Results are written under `--result_dir`. Dumped images use sanitized filenames to avoid path issues.

## Troubleshooting

- Missing masks: verify `images/` and `annotations/` names match.
- Import errors: ensure you installed the package from repo root.
- CUDA errors: use `--amp` only when CUDA is available.
