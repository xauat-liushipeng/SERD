# Semantic-Edge Response Decoding of SAM3 for Zero-Shot Crack Segmentation



## Metrics

All summary metrics are macro-averaged over images.

- `crack_iou`: foreground (crack) IoU.
- `miou`: mean of crack IoU and background IoU.
- `f1`: crack-class F1 score.
- `precision`: crack-class precision (`P`).
- `recall`: crack-class recall (`R`).
- `par`: predicted crack area divided by GT crack area (`PAR`).

`background_iou` is also written for auditability.

## Installation

Use Python 3.12, a CUDA-enabled PyTorch build, and the official SAM3 repository.

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3
```

Place the authorized SAM3 checkpoint at `checkpoints/sam3.pt`, or pass its path with `--sam_checkpoint`.

## Dataset layout

By default, `--data_root` points to a `datasets` directory next to this repository. The expected structure is:

```text
datasets/
├── Crack500/test_img, test_lab
├── CamCrack789/test_img, test_lab
├── CrackMap/test_img, test_lab
├── DeepCrack/test_img, test_lab
├── TUT/test_img, test_lab
└── omnicrack30k/images/test, annotations/test
```

All built-in masks use nonzero pixels as crack pixels. A custom dataset can use black cracks with `--gt_positive zero`.

Check all paths and image/mask pairs without loading the model:

```bash
python eval_SERD.py --datasets all --data_root /path/to/datasets --preflight
```

## Evaluation

Evaluate SERD on every dataset with the paper defaults (`tau=0.40`, `alpha=1.0`, Sobel edge prior):

```bash
python eval_SERD.py --datasets all --data_root /path/to/datasets \
  --sam_checkpoint /path/to/sam3.pt --sam3_repo /path/to/sam3
```

Evaluate the original SAM3 baseline:

```bash
python eval_SAM3.py --datasets all --data_root /path/to/datasets \
  --sam_checkpoint /path/to/sam3.pt --sam3_repo /path/to/sam3 \
  --sam_confidence_threshold 0.50
```

For a custom dataset:

```bash
python eval_SERD.py --image_dir /path/to/images --mask_dir /path/to/masks \
  --dataset_name MyDataset --gt_positive nonzero \
  --sam_checkpoint /path/to/sam3.pt --sam3_repo /path/to/sam3
```

Each evaluator writes per-dataset `per_image.csv`, `summary.csv`, `summary.json`, and a top-level `all_summary.csv`. Prediction masks are only written when `--save_masks` is supplied. Individual failures are logged and skipped unless `--fail_on_error` is supplied.

## Evaluation Results

| Method   | Crack IoU | mIoU  |  F1   |   R   |   P   |  PAR  |
| -------- | :-------: | :---: | :---: | :---: | :---: | :---: |
| SAM3     |   56.51   | 77.43 | 66.96 | 70.38 | 68.35 | 0.934 |
| SERD     |   61.14   | 79.67 | 72.62 | 78.01 | 73.05 | 1.164 |
| $\Delta$ |   +4.63   | +2.24 | +5.66 | +7.63 | +4.70 |   -   |
