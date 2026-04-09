"""
Dataloader factory — selects the appropriate dataloader based on the ROBOT_TYPE
environment variable.

ROBOT_TYPE values:
    corobot  — DataLoaderCoRobot (requires a2d_sdk)
    x2robot  — DataLoaderX2Robot (requires x2robot bridge server)
    mock     — DataLoaderMock (synthetic or file-based images, no hardware)
    (unset)  — auto-detect: try corobot → x2robot → mock
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base_dataloader import BaseDataloader

logger = logging.getLogger(__name__)


def create_robot_dataloader() -> tuple[Optional[BaseDataloader], Optional[str]]:
    """
    Create the appropriate dataloader based on ROBOT_TYPE env var.

    Returns:
        (dataloader, warning_message)
        - dataloader may be None if nothing works (chat-only mode)
        - warning_message is set when falling back from the preferred choice
    """
    robot_type = os.environ.get("ROBOT_TYPE", "").strip().lower()

    if robot_type == "corobot":
        return _try_corobot()
    elif robot_type == "x2robot":
        return _try_x2robot()
    elif robot_type == "mock":
        return _try_mock()
    else:
        return _auto_detect()


def _try_corobot() -> tuple[Optional[BaseDataloader], Optional[str]]:
    try:
        from .dataloader_corobot import DataLoaderCoRobot
        return DataLoaderCoRobot(base_url="http://localhost:8765"), None
    except Exception as exc:
        logger.warning("CoRobot dataloader unavailable (%s), falling back to mock", exc)
        return _try_mock_with_warning(f"CoRobot unavailable: {exc}")


def _try_x2robot() -> tuple[Optional[BaseDataloader], Optional[str]]:
    try:
        from .dataloader_x2robot import DataLoaderX2Robot
        bridge_url = os.environ.get("X2ROBOT_BRIDGE_URL", "").strip() or None
        return DataLoaderX2Robot(fmt="jpeg", base_url=bridge_url), None
    except Exception as exc:
        logger.warning("x2robot dataloader unavailable (%s), falling back to mock", exc)
        return _try_mock_with_warning(f"x2robot unavailable: {exc}")


def _try_mock() -> tuple[Optional[BaseDataloader], Optional[str]]:
    try:
        from .dataloader_mock import DataLoaderMock
        return DataLoaderMock(fmt="jpeg"), None
    except Exception as exc:
        logger.error("MockDataLoader also unavailable: %s", exc)
        return None, f"All dataloaders unavailable; chat-only mode: {exc}"


def _try_mock_with_warning(reason: str) -> tuple[Optional[BaseDataloader], Optional[str]]:
    try:
        from .dataloader_mock import DataLoaderMock
        mock = DataLoaderMock(fmt="jpeg")
        return mock, f"{reason}; using MockDataLoader"
    except Exception as mock_exc:
        warning = f"{reason}; MockDataLoader also failed: {mock_exc}; chat-only mode"
        logger.error(warning)
        return None, warning


def _auto_detect() -> tuple[Optional[BaseDataloader], Optional[str]]:
    """Try corobot → x2robot → mock in order."""
    loader, warning = _try_corobot()
    if loader is not None and warning is None:
        return loader, None

    loader, warning = _try_x2robot()
    if loader is not None and warning is None:
        return loader, None

    return _try_mock_with_warning("No hardware dataloader available (auto-detect)")
