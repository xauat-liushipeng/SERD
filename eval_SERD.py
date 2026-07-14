from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval_common import (
    DATASET_LAYOUTS,
    add_shared_arguments,
    align_and_resize,
    binary_metrics,
    build_dataset_jobs,
    find_mask,
    imread_unicode,
    list_images,
    load_rgb,
    normalize01,
    preflight_jobs,
    save_mask,
    summarize_rows,
    write_csv,
    write_json,
)
from sam3_runner import Sam3Runner


METHOD = "SERD"


def edge_prior(gray: np.ndarray, mode: str) -> np.ndarray:
    gray01 = normalize01(gray.astype(np.float32))
    if mode == "sobel":
        sx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
        return normalize01(np.sqrt(sx * sx + sy * sy))
    if mode == "canny":
        return (cv2.Canny((gray01 * 255).astype(np.uint8), 50, 150) > 0).astype(np.float32)
    if mode == "log":
        blurred = cv2.GaussianBlur(gray01, (0, 0), 1.2)
        return normalize01(np.abs(cv2.Laplacian(blurred, cv2.CV_32F)))
    if mode == "tophat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx((gray01 * 255).astype(np.uint8), cv2.MORPH_BLACKHAT, kernel)
        return normalize01(blackhat)
    raise ValueError(f"unknown edge mode: {mode}")


def serd_decode(semantic: np.ndarray, edge: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    response = normalize01(semantic)
    calibrated = normalize01(response * (1.0 + float(alpha) * normalize01(edge)))
    return calibrated >= float(tau)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SERD on all crack datasets.")
    add_shared_arguments(parser, "outputs/eval_SERD")
    parser.add_argument("--tau", type=float, default=0.40, help="SERD decision threshold.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Edge calibration strength.")
    parser.add_argument("--edge_mode", default="sobel", choices=["sobel", "canny", "log", "tophat"])
    return parser.parse_args()


def evaluate_dataset(
    job: dict[str, Any], runner: Sam3Runner, args: argparse.Namespace, out_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    images = list_images(Path(job["image_dir"]))
    if args.limit is not None:
        images = images[: args.limit]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, image_path in enumerate(images, 1):
        try:
            gt_path = find_mask(Path(job["mask_dir"]), image_path.stem)
            if gt_path is None:
                raise FileNotFoundError(f"mask not found for {image_path.name}")
            image_rgb = load_rgb(image_path)
            gt_gray = imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_gray is None:
                raise FileNotFoundError(f"failed to read mask: {gt_path}")
            image_rgb, gt_gray, trace = align_and_resize(image_rgb, gt_gray, args.max_image_side)
            gt = gt_gray > 0 if job["gt_positive"] == "nonzero" else gt_gray == 0
            semantic = runner.predict_semantic(image_rgb, args.text_prompt)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            pred = serd_decode(semantic, edge_prior(gray, args.edge_mode), args.tau, args.alpha)
            row = {
                "dataset": job["name"],
                "method": METHOD,
                "image": str(image_path),
                "mask": str(gt_path),
                **binary_metrics(pred, gt),
                "tau": float(args.tau),
                "alpha": float(args.alpha),
                "edge_mode": args.edge_mode,
                **trace,
            }
            rows.append(row)
            if args.save_masks:
                save_mask(out_dir / "pred_masks" / f"{image_path.stem}.png", pred)
            if index == 1 or index % args.progress_every == 0 or index == len(images):
                current = summarize_rows(rows, job["name"], METHOD)
                print(
                    f"[{job['name']}] {index}/{len(images)} crack_iou={row['crack_iou']:.4f} "
                    f"running_miou={current['miou']:.4f}",
                    flush=True,
                )
        except Exception as exc:
            failures.append({"dataset": job["name"], "image": str(image_path), "error": repr(exc)})
            print(f"[{job['name']}] failed {image_path.name}: {exc}", flush=True)
            if args.fail_on_error:
                raise
        finally:
            gc.collect()
            if args.cuda_cleanup_every > 0 and index % args.cuda_cleanup_every == 0:
                runner.release_cuda_cache()

    summary = summarize_rows(rows, job["name"], METHOD)
    summary.update(
        {
            "total_images": len(images),
            "failures": len(failures),
            "gt_positive": job["gt_positive"],
            "tau": float(args.tau),
            "alpha": float(args.alpha),
            "edge_mode": args.edge_mode,
        }
    )
    write_csv(out_dir / "per_image.csv", rows)
    write_csv(out_dir / "failures.csv", failures)
    write_csv(out_dir / "summary.csv", [summary])
    write_json(out_dir / "summary.json", {"summary": summary, "samples": rows, "failures": failures})
    return rows, failures, summary


def main() -> None:
    args = parse_args()
    if args.datasets.strip().casefold() == "list":
        print("\n".join(DATASET_LAYOUTS))
        return
    jobs = build_dataset_jobs(args)
    if args.preflight:
        report = preflight_jobs(jobs)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not all(item["image_dir_exists"] and item["mask_dir_exists"] for item in report):
            raise SystemExit(2)
        return

    runner = Sam3Runner(
        checkpoint=args.sam_checkpoint,
        sam3_repo=args.sam3_repo,
        device=args.device,
        confidence_threshold=0.0,
    )
    root_out = Path(args.out_dir).expanduser().resolve()
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        rows, failures, summary = evaluate_dataset(job, runner, args, root_out / job["name"])
        all_rows.extend(rows)
        all_failures.extend(failures)
        summaries.append(summary)

    overall = summarize_rows(all_rows, "ALL", METHOD)
    overall.update(
        {
            "total_images": sum(item["total_images"] for item in summaries),
            "failures": len(all_failures),
            "tau": float(args.tau),
            "alpha": float(args.alpha),
            "edge_mode": args.edge_mode,
        }
    )
    write_csv(root_out / "all_summary.csv", summaries + [overall])
    write_csv(root_out / "all_per_image.csv", all_rows)
    write_csv(root_out / "all_failures.csv", all_failures)
    write_json(root_out / "all_summary.json", {"datasets": summaries, "summary": overall})
    print(json.dumps({"datasets": summaries, "summary": overall}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

