from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Paths are relative to --data_root. The defaults match the layouts used in the
# paper experiments while remaining portable across machines.
DATASET_LAYOUTS: dict[str, dict[str, str]] = {
    "Crack500": {
        "image_dir": "Crack500/test_img",
        "mask_dir": "Crack500/test_lab",
        "gt_positive": "nonzero",
    },
    "CamCrack789": {
        "image_dir": "CamCrack789/test_img",
        "mask_dir": "CamCrack789/test_lab",
        "gt_positive": "nonzero",
    },
    "CrackMap": {
        "image_dir": "CrackMap/test_img",
        "mask_dir": "CrackMap/test_lab",
        "gt_positive": "nonzero",
    },
    "DeepCrack": {
        "image_dir": "DeepCrack/test_img",
        "mask_dir": "DeepCrack/test_lab",
        "gt_positive": "nonzero",
    },
    "TUT": {
        "image_dir": "TUT/test_img",
        "mask_dir": "TUT/test_lab",
        "gt_positive": "nonzero",
    },
    "OmniCrack30k": {
        "image_dir": "omnicrack30k/images/test",
        "mask_dir": "omnicrack30k/annotations/test",
        "gt_positive": "nonzero",
    },
}

DATASET_ALIASES = {name.casefold(): name for name in DATASET_LAYOUTS}
DATASET_ALIASES.update(
    {
        "crack-500": "Crack500",
        "camcrack": "CamCrack789",
        "camcrack-789": "CamCrack789",
        "crackmap": "CrackMap",
        "crack-map": "CrackMap",
        "deep-crack": "DeepCrack",
        "omni": "OmniCrack30k",
        "omnicrack": "OmniCrack30k",
        "omnicrack-30k": "OmniCrack30k",
    }
)


def default_data_root() -> str:
    return os.environ.get("SERD_DATA_ROOT", str(PROJECT_DIR.parent / "datasets"))


def default_checkpoint() -> str:
    return os.environ.get("SERD_SAM_CHECKPOINT", str(PROJECT_DIR / "checkpoints" / "sam3.pt"))


def default_sam3_repo() -> str:
    return os.environ.get("SERD_SAM3_REPO", str(PROJECT_DIR / "third_party" / "sam3"))


def add_shared_arguments(parser: argparse.ArgumentParser, default_out_dir: str) -> None:
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated built-in dataset names, 'all', or 'list'.",
    )
    parser.add_argument("--data_root", default=default_data_root(), help="Root containing the built-in datasets.")
    parser.add_argument("--image_dir", default="", help="Custom image directory; overrides --datasets.")
    parser.add_argument("--mask_dir", default="", help="Custom GT mask directory for --image_dir.")
    parser.add_argument("--dataset_name", default="custom", help="Output name for a custom dataset.")
    parser.add_argument(
        "--gt_positive",
        default="nonzero",
        choices=["nonzero", "zero"],
        help="Pixels interpreted as crack in custom GT masks.",
    )
    parser.add_argument("--sam_checkpoint", default=default_checkpoint())
    parser.add_argument("--sam3_repo", default=default_sam3_repo(), help="SAM3 source checkout; may be empty if installed.")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu.")
    parser.add_argument("--text_prompt", default="crack")
    parser.add_argument("--max_image_side", type=int, default=1024, help="0 keeps native resolution.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images per dataset.")
    parser.add_argument("--out_dir", default=default_out_dir)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--cuda_cleanup_every", type=int, default=25)
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check dataset paths and image/mask pairing without loading SAM3.",
    )


def parse_dataset_names(spec: str) -> list[str]:
    if spec.strip().casefold() == "all":
        return list(DATASET_LAYOUTS)
    names: list[str] = []
    unknown: list[str] = []
    for raw in (item.strip() for item in spec.split(",")):
        if not raw:
            continue
        canonical = DATASET_ALIASES.get(raw.casefold())
        if canonical is None:
            unknown.append(raw)
        elif canonical not in names:
            names.append(canonical)
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; valid={list(DATASET_LAYOUTS)}")
    if not names:
        raise ValueError("--datasets did not select any dataset")
    return names


def build_dataset_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.image_dir:
        if not args.mask_dir:
            raise ValueError("--mask_dir is required with --image_dir")
        return [
            {
                "name": args.dataset_name,
                "image_dir": Path(args.image_dir).expanduser().resolve(),
                "mask_dir": Path(args.mask_dir).expanduser().resolve(),
                "gt_positive": args.gt_positive,
            }
        ]

    root = Path(args.data_root).expanduser().resolve()
    jobs = []
    for name in parse_dataset_names(args.datasets):
        layout = DATASET_LAYOUTS[name]
        jobs.append(
            {
                "name": name,
                "image_dir": root / layout["image_dir"],
                "mask_dir": root / layout["mask_dir"],
                "gt_positive": layout["gt_positive"],
            }
        )
    return jobs


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_rgb(path: Path) -> np.ndarray:
    image = imread_unicode(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    return sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def candidate_mask_stems(stem: str) -> list[str]:
    stems = [stem]
    if stem.startswith("image-"):
        stems.append("target-" + stem[len("image-") :])
    if stem.startswith("target-"):
        stems.append("image-" + stem[len("target-") :])
    if stem.endswith("_img"):
        stems.append(stem[:-4])
    return list(dict.fromkeys(stems))


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for candidate in candidate_mask_stems(stem):
        for suffix in MASK_EXTENSIONS:
            path = mask_dir / f"{candidate}{suffix}"
            if path.exists():
                return path
    return None


def align_and_resize(
    image_rgb: np.ndarray,
    gt_gray: np.ndarray,
    max_image_side: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    original_image_shape = image_rgb.shape[:2]
    original_gt_shape = gt_gray.shape[:2]
    gt_resized_to_image = original_gt_shape != original_image_shape
    if gt_resized_to_image:
        gt_gray = cv2.resize(
            gt_gray,
            (original_image_shape[1], original_image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    scale = 1.0
    if max_image_side > 0 and max(original_image_shape) > max_image_side:
        scale = max_image_side / float(max(original_image_shape))
        width = max(1, int(round(original_image_shape[1] * scale)))
        height = max(1, int(round(original_image_shape[0] * scale)))
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        gt_gray = cv2.resize(gt_gray, (width, height), interpolation=cv2.INTER_NEAREST)

    trace = {
        "original_height": int(original_image_shape[0]),
        "original_width": int(original_image_shape[1]),
        "original_gt_height": int(original_gt_shape[0]),
        "original_gt_width": int(original_gt_shape[1]),
        "gt_resized_to_image": bool(gt_resized_to_image),
        "eval_height": int(image_rgb.shape[0]),
        "eval_width": int(image_rgb.shape[1]),
        "resize_scale": float(scale),
    }
    return image_rgb, gt_gray, trace


def normalize01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi <= lo + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - lo) / (hi - lo)


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask).astype(bool)
    if value.shape == shape:
        return value
    return cv2.resize(value.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def union_masks(masks: list[np.ndarray], shape: tuple[int, int], max_masks: int = 0) -> np.ndarray:
    selected = masks[:max_masks] if max_masks > 0 else masks
    output = np.zeros(shape, dtype=bool)
    for mask in selected:
        output |= resize_bool(mask, shape)
    return output


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    if pred_bool.shape != gt_bool.shape:
        raise ValueError(f"prediction/GT shape mismatch: {pred_bool.shape} vs {gt_bool.shape}")

    tp = np.logical_and(pred_bool, gt_bool).sum(dtype=np.float64)
    pred_area = pred_bool.sum(dtype=np.float64)
    gt_area = gt_bool.sum(dtype=np.float64)
    crack_union = np.logical_or(pred_bool, gt_bool).sum(dtype=np.float64)
    crack_iou = float(tp / max(crack_union, 1.0))
    precision = float(tp / max(pred_area, 1.0))
    recall = float(tp / max(gt_area, 1.0))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))

    bg_pred = ~pred_bool
    bg_gt = ~gt_bool
    bg_intersection = np.logical_and(bg_pred, bg_gt).sum(dtype=np.float64)
    bg_union = np.logical_or(bg_pred, bg_gt).sum(dtype=np.float64)
    background_iou = float(bg_intersection / max(bg_union, 1.0))

    return {
        "crack_iou": crack_iou,
        "background_iou": background_iou,
        "miou": float((crack_iou + background_iou) / 2.0),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "par": float(pred_area / max(gt_area, 1.0)),
    }


METRIC_KEYS = ("crack_iou", "background_iou", "miou", "f1", "precision", "recall", "par")


def summarize_rows(rows: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset": dataset, "method": method, "evaluated": len(rows)}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.asarray(mask).astype(np.uint8) * 255)
    if not ok:
        raise OSError(f"failed to encode mask: {path}")
    encoded.tofile(path)


def preflight_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for job in jobs:
        image_dir = Path(job["image_dir"])
        mask_dir = Path(job["mask_dir"])
        images = list_images(image_dir) if image_dir.is_dir() else []
        matched = sum(1 for image in images if mask_dir.is_dir() and find_mask(mask_dir, image.stem) is not None)
        report.append(
            {
                "dataset": job["name"],
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "image_dir_exists": image_dir.is_dir(),
                "mask_dir_exists": mask_dir.is_dir(),
                "images": len(images),
                "matched_masks": matched,
                "missing_masks": len(images) - matched,
                "gt_positive": job["gt_positive"],
            }
        )
    return report

