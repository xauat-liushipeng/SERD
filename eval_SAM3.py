from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import cv2

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
    preflight_jobs,
    save_mask,
    summarize_rows,
    union_masks,
    write_csv,
    write_json,
)
from sam3_runner import Sam3Runner


METHOD = "SAM3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text-prompted SAM3 on all crack datasets.")
    add_shared_arguments(parser, "outputs/eval_SAM3")
    parser.add_argument("--sam_confidence_threshold", type=float, default=0.50)
    parser.add_argument("--max_masks", type=int, default=0, help="Union top N masks; 0 unions all returned masks.")
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
            masks, scores = runner.predict_text(image_rgb, args.text_prompt)
            pred = union_masks(masks, gt.shape, args.max_masks)
            row = {
                "dataset": job["name"],
                "method": METHOD,
                "image": str(image_path),
                "mask": str(gt_path),
                **binary_metrics(pred, gt),
                "num_masks": len(masks),
                "scores": json.dumps(scores, ensure_ascii=False),
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
    summary.update({"total_images": len(images), "failures": len(failures), "gt_positive": job["gt_positive"]})
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
        confidence_threshold=args.sam_confidence_threshold,
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
    overall.update({"total_images": sum(item["total_images"] for item in summaries), "failures": len(all_failures)})
    write_csv(root_out / "all_summary.csv", summaries + [overall])
    write_csv(root_out / "all_per_image.csv", all_rows)
    write_csv(root_out / "all_failures.csv", all_failures)
    write_json(root_out / "all_summary.json", {"datasets": summaries, "summary": overall})
    print(json.dumps({"datasets": summaries, "summary": overall}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

