"""
Mock DataLoader — drop-in replacement for DataLoaderCoRobot.

Supports two modes controlled by environment variables:

1. Real images (recommended for VLM testing):
   Set ROBOCLAW_MOCK_IMAGE_DIR to a directory containing .jpg/.png files.
   Each image is split into left/center/right thirds to simulate the
   three camera views (hand_left / head / hand_right), cycling through
   images on successive calls.

2. Synthetic images (fallback):
   If no image directory is set, generates colored gradient placeholders.

Set ROBOCLAW_TEXT_ONLY=1 to skip all image injection (for text-only models).
"""

from __future__ import annotations

import base64
import glob
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

from .base_dataloader import BaseDataloader
from agent_demo.types.machine_layer import A2DData

logger = logging.getLogger(__name__)

IMG_HEIGHT = 480
IMG_WIDTH = 640


class _StubSlam:
    def navigate_to_pose(self, x: int, y: int, theta: int) -> None:
        logger.info("[MockSLAM] navigate_to_pose(x=%d, y=%d, θ=%d) — no-op", x, y, theta)


class _StubCamera:
    pass


class _StubRobot:
    def shutdown(self) -> None:
        pass


def _generate_camera_image(label: str, width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> np.ndarray:
    """Create a synthetic BGR image with a color gradient and label."""
    color_map = {
        "head": ((180, 120, 60), (60, 120, 180)),
        "hand_left": ((60, 180, 120), (120, 60, 180)),
        "hand_right": ((120, 60, 180), (180, 180, 60)),
    }
    top_color, bot_color = color_map.get(label, ((128, 128, 128), (64, 64, 64)))

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        alpha = row / max(height - 1, 1)
        for c in range(3):
            img[row, :, c] = int(top_color[c] * (1 - alpha) + bot_color[c] * alpha)

    ts_text = time.strftime("%H:%M:%S")
    cv2.putText(img, f"[MOCK] {label}", (20, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, ts_text, (20, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return img


def _discover_images(directory: str) -> list[str]:
    """Find all jpg/png images in a directory, sorted by name."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    files.sort()
    return files


def _split_into_three_views(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a single photo into left / center / right thirds to simulate 3 cameras."""
    h, w = img.shape[:2]
    third = w // 3
    left = img[:, :third]
    center = img[:, third : third * 2]
    right = img[:, third * 2 :]
    return left, center, right


class DataLoaderMock(BaseDataloader):
    """
    Mock DataLoader that produces images for the Agent pipeline.
    No hardware dependencies — always succeeds.
    """

    def __init__(
        self,
        fmt: Literal["jpeg", "png", "jpg"] = "jpeg",
        base_url: str | None = None,
        env_config: dict | None = None,
    ):
        self._robot = _StubRobot()
        self._slam = _StubSlam()
        self._camera = _StubCamera()
        self._frame_id: int = 0
        self._format: Literal["jpeg", "png", "jpg"] = fmt
        self._ext = ".jpg" if fmt in ("jpeg", "jpg") else ".png"

        self._image_files: list[str] = []
        self._image_cursor: int = 0

        img_dir = os.environ.get("ROBOCLAW_MOCK_IMAGE_DIR", "").strip()
        if img_dir and os.path.isdir(img_dir):
            self._image_files = _discover_images(img_dir)
            if self._image_files:
                logger.info(
                    "[MockDataLoader] initialized with %d real images from %s",
                    len(self._image_files), img_dir,
                )
            else:
                logger.warning("[MockDataLoader] ROBOCLAW_MOCK_IMAGE_DIR=%s has no images, using synthetic", img_dir)
        else:
            logger.info("[MockDataLoader] initialized (synthetic mode, format=%s)", fmt)

    @property
    def robot(self):
        return self._robot

    @property
    def slam(self):
        return self._slam

    @property
    def camera(self):
        return self._camera

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def frame_id_auto_plus(self) -> int:
        fid = self._frame_id
        self._frame_id = (self._frame_id + 1) % 1_000_000
        return fid

    def shutdown(self) -> None:
        logger.info("[MockDataLoader] shutdown()")

    _MAX_HEIGHT = 480

    _REPEAT_PER_IMAGE = int(os.environ.get("ROBOCLAW_MOCK_IMAGE_REPEAT", "5"))

    def _next_real_image(self) -> np.ndarray | None:
        """Load the next image from the directory, cycling. Each image repeats _REPEAT_PER_IMAGE times."""
        if not self._image_files:
            return None
        idx = (self._image_cursor // self._REPEAT_PER_IMAGE) % len(self._image_files)
        path = self._image_files[idx]
        self._image_cursor += 1
        img = cv2.imread(path)
        if img is None:
            logger.warning("[MockDataLoader] failed to read %s", path)
            return None
        h, w = img.shape[:2]
        if h > self._MAX_HEIGHT:
            scale = self._MAX_HEIGHT / h
            img = cv2.resize(img, (int(w * scale), self._MAX_HEIGHT))
        logger.info("[MockDataLoader] loaded %s (%dx%d)", Path(path).name, img.shape[1], img.shape[0])
        return img

    async def get_latest_concatenate_image_base64(self, need_save: bool = False) -> Optional[A2DData]:
        if os.environ.get("ROBOCLAW_TEXT_ONLY", "").strip() in ("1", "true", "yes"):
            logger.debug("[MockDataLoader] ROBOCLAW_TEXT_ONLY is set, skipping image generation")
            return None

        real_img = self._next_real_image()
        if real_img is not None:
            left_img, head_img, right_img = _split_into_three_views(real_img)
        else:
            head_img = _generate_camera_image("head")
            left_img = _generate_camera_image("hand_left")
            right_img = _generate_camera_image("hand_right")

        fid = self.frame_id_auto_plus
        ts_ns = int(time.time() * 1e9)

        a2d = A2DData(
            frame_id=fid,
            image_type=self._format,
            image_ts=ts_ns,
            head_image=head_img,
            left_wrist_image=left_img,
            right_wrist_image=right_img,
        )

        concat = self._build_concatenated(a2d, need_save)
        if concat is None:
            return None

        return a2d

    # ---- Internal helpers (same logic as DataLoaderCoRobot) ----

    def _build_concatenated(self, a2d: A2DData, need_save: bool) -> Optional[str]:
        if a2d.head_image is None or a2d.left_wrist_image is None or a2d.right_wrist_image is None:
            return None

        min_h = min(a2d.left_wrist_image.shape[0], a2d.head_image.shape[0], a2d.right_wrist_image.shape[0])
        left_r = self._resize_h(a2d.left_wrist_image, min_h)
        head_r = self._resize_h(a2d.head_image, min_h)
        right_r = self._resize_h(a2d.right_wrist_image, min_h)

        left_a = self._annotate(left_r, "Left Wrist")
        head_a = self._annotate(head_r, "Head")
        right_a = self._annotate(right_r, "Right Wrist")

        concatenated = cv2.hconcat([left_a, head_a, right_a])
        a2d.concatenated_image = concatenated

        ok, encoded = cv2.imencode(self._ext, concatenated)
        if not ok:
            logger.error("[MockDataLoader] image encode failed")
            return None

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        a2d.concatenated_image_base64 = b64

        if need_save:
            save_dir = "./img_dir"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"mock_img_{a2d.frame_id}{self._ext}")
            cv2.imwrite(path, concatenated)
            logger.info("[MockDataLoader] saved %s", path)

        return b64

    @staticmethod
    def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
        oh, ow = img.shape[:2]
        scale = h / oh
        return cv2.resize(img, (int(ow * scale), h))

    @staticmethod
    def _annotate(img: np.ndarray, label: str, top_border: int = 15, side_border: int = 3) -> np.ndarray:
        bordered = cv2.copyMakeBorder(img, top=top_border, bottom=0, left=side_border, right=side_border,
                                      borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(label, font, 0.5, 1)[0]
        tx = (bordered.shape[1] - ts[0]) // 2
        ty = (top_border + ts[1]) // 2
        cv2.putText(bordered, label, (tx, ty), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return bordered
