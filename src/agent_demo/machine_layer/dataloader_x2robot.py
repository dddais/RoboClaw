"""
x2robot DataLoader - Fetches three-camera images from x2robot_bridge_server via HTTP.

Provides the same A2DData interface as DataLoaderCoRobot so the Agent pipeline
(ImgActAgent, MemoryManager) works without modification.

Environment variables:
    X2ROBOT_BRIDGE_URL  - Bridge server base URL (default: http://localhost:8766)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Literal, Optional

import cv2
import httpx
import numpy as np

from .base_dataloader import BaseDataloader
from agent_demo.types.machine_layer import A2DData

logger = logging.getLogger(__name__)


class _StubSlam:
    def navigate_to_pose(self, x: int, y: int, theta: int) -> None:
        logger.info("[x2robot] navigate_to_pose(x=%d, y=%d, θ=%d) — not supported", x, y, theta)


class _StubCamera:
    pass


class _StubRobot:
    def shutdown(self) -> None:
        pass


class DataLoaderX2Robot(BaseDataloader):
    """
    x2robot DataLoader - Fetches images from x2robot_bridge_server HTTP API.
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

        self._bridge_url = (
            base_url
            or os.environ.get("X2ROBOT_BRIDGE_URL", "").strip()
            or ""
        )
        if not self._bridge_url:
            self._bridge_url = "http://localhost:8766"
            logger.warning(
                "[x2robot] X2ROBOT_BRIDGE_URL not set, using default %s "
                "(only valid for local mock testing). "
                "For real deployment, set X2ROBOT_BRIDGE_URL=http://<robot_ip>:8766",
                self._bridge_url,
            )

        logger.info(
            "[x2robot] DataLoaderX2Robot initialized, bridge_url=%s, format=%s",
            self._bridge_url, fmt,
        )

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
        logger.info("[x2robot] DataLoaderX2Robot.shutdown()")

    async def get_latest_concatenate_image_base64(self, need_save: bool = False) -> Optional[A2DData]:
        """Fetch three camera images from bridge, concatenate, and return A2DData."""
        try:
            left_img, head_img, right_img = await self._fetch_images()
        except Exception as e:
            logger.error("[x2robot] Failed to fetch images from bridge: %s", e)
            return None

        if left_img is None or head_img is None or right_img is None:
            logger.warning("[x2robot] Incomplete camera data from bridge")
            return None

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

        b64 = self._build_concatenated(a2d, need_save)
        if b64 is None:
            return None

        return a2d

    async def _fetch_images(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        url = f"{self._bridge_url}/cameras/latest"
        timeout = httpx.Timeout(timeout=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.json()

        if not body.get("success"):
            logger.warning("[x2robot] Bridge returned failure: %s", body.get("message"))
            return None, None, None

        data = body.get("data", {})

        left = self._decode_b64_image(data.get("left_wrist"))
        head = self._decode_b64_image(data.get("head"))
        right = self._decode_b64_image(data.get("right_wrist"))

        return left, head, right

    @staticmethod
    def _decode_b64_image(b64_str: Optional[str]) -> Optional[np.ndarray]:
        if not b64_str:
            return None
        try:
            raw = base64.b64decode(b64_str)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.warning("[x2robot] Failed to decode base64 image: %s", e)
            return None

    def _build_concatenated(self, a2d: A2DData, need_save: bool) -> Optional[str]:
        if a2d.head_image is None or a2d.left_wrist_image is None or a2d.right_wrist_image is None:
            return None

        min_h = min(
            a2d.left_wrist_image.shape[0],
            a2d.head_image.shape[0],
            a2d.right_wrist_image.shape[0],
        )
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
            logger.error("[x2robot] Image encode failed")
            return None

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        a2d.concatenated_image_base64 = b64

        if need_save:
            save_dir = "./img_dir"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"x2robot_img_{a2d.frame_id}{self._ext}")
            cv2.imwrite(path, concatenated)
            logger.info("[x2robot] Saved %s", path)

        return b64

    @staticmethod
    def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
        oh, ow = img.shape[:2]
        scale = h / oh
        return cv2.resize(img, (int(ow * scale), h))

    @staticmethod
    def _annotate(img: np.ndarray, label: str, top_border: int = 15, side_border: int = 3) -> np.ndarray:
        bordered = cv2.copyMakeBorder(
            img, top=top_border, bottom=0, left=side_border, right=side_border,
            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(label, font, 0.5, 1)[0]
        tx = (bordered.shape[1] - ts[0]) // 2
        ty = (top_border + ts[1]) // 2
        cv2.putText(bordered, label, (tx, ty), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return bordered
