from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Sam3Runner:
    """Small adapter around the official SAM3 image model used by the evaluators."""

    def __init__(
        self,
        checkpoint: str | Path,
        sam3_repo: str | Path = "",
        device: str | None = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError("SAM3 evaluation requires PyTorch and Pillow") from exc

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        repo_text = str(sam3_repo).strip()
        if repo_text:
            repo = Path(repo_text).expanduser().resolve()
            if repo.is_dir() and str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError(
                "Could not import SAM3. Install the official SAM3 package or pass --sam3_repo."
            ) from exc

        self._torch = torch
        self._image_class = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            load_from_HF=False,
            enable_inst_interactivity=False,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=float(confidence_threshold),
        )

    def _context(self):
        if str(self.device).startswith("cuda"):
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    def _clear_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        try:
            self.processor.reset_all_prompts(state)
        finally:
            state.clear()

    def release_cuda_cache(self) -> None:
        if str(self.device).startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def predict_text(self, image_rgb: np.ndarray, prompt: str) -> tuple[list[np.ndarray], list[float]]:
        state: dict[str, Any] | None = None
        try:
            with self._context():
                state = self.processor.set_image(self._image_class.fromarray(image_rgb.astype(np.uint8)))
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            scores = output.get("scores")
            if masks is None:
                return [], []
            masks_np = masks.detach().cpu().numpy()
            scores_np = (
                scores.detach().float().cpu().numpy()
                if scores is not None
                else np.ones((len(masks_np),), dtype=np.float32)
            )
            result_masks = [np.squeeze(mask).astype(bool) for mask in masks_np]
            result_scores = [float(scores_np[index]) for index in range(len(result_masks))]
            order = np.argsort(np.asarray(result_scores, dtype=np.float32))[::-1]
            return [result_masks[int(i)] for i in order], [result_scores[int(i)] for i in order]
        finally:
            self._clear_state(state)

    def predict_semantic(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        """Return the normalized SAM3 text-conditioned semantic response."""
        state: dict[str, Any] | None = None
        try:
            with self._context():
                state = self.processor.set_image(self._image_class.fromarray(image_rgb.astype(np.uint8)))
                text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
                state["backbone_out"].update(text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = self.model._get_dummy_prompt()
                output = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=self.processor.find_stage,
                    geometric_prompt=state["geometric_prompt"],
                    find_target=None,
                )
            semantic = output.get("semantic_seg")
            if semantic is None:
                raise KeyError("SAM3 output does not contain semantic_seg")
            score = semantic[0, 0].float().sigmoid().detach().cpu().numpy()
            if score.shape != image_rgb.shape[:2]:
                score = cv2.resize(
                    score.astype(np.float32),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            return score.astype(np.float32)
        finally:
            self._clear_state(state)

