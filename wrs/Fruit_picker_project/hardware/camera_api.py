# hardware/camera_api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FruitTarget:
    xyz: tuple[float, float, float]
    confidence: float


class CameraAPI:
    """相机适配层。

    优先复用 `pick/pepper_picker.py` 中的真实相机与YOLO识别链路；
    若外部采摘器尚未注入，则降级为由 `scan_and_detect` 兼容的返回值包装。
    """

    def __init__(self, model_path: str, device: int = 0, confidence_threshold: float = 0.8, backend: Any | None = None):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        print(f"[硬件模块] 深度相机初始化完成，模型: {self.model_path}, device={self.device}")

    def set_backend(self, backend: Any) -> None:
        self.backend = backend

    def _wrap_detection(self, det) -> list[FruitTarget]:
        if det is None:
            return []
        if isinstance(det, list):
            wrapped: list[FruitTarget] = []
            for item in det:
                if isinstance(item, FruitTarget):
                    wrapped.append(item)
                elif isinstance(item, dict) and "xyz" in item:
                    conf = float(item.get("confidence", 1.0))
                    if conf >= self.confidence_threshold:
                        wrapped.append(FruitTarget(tuple(item["xyz"]), conf))
            return wrapped
        if isinstance(det, np.ndarray):
            if det.size >= 3:
                return [FruitTarget((float(det[0]), float(det[1]), float(det[2])), 1.0)]
            return []
        if isinstance(det, (tuple, list)) and len(det) >= 3:
            return [FruitTarget((float(det[0]), float(det[1]), float(det[2])), 1.0)]
        return []

    def detect_targets(self) -> list[FruitTarget]:
        print("[相机 API] 正在调用真实相机/YOLO采摘链路...")
        if self.backend is None:
            print("[相机 API] 未注入真实采摘器，无法进行真实检测。")
            return []
        try:
            det = self.backend.scan_and_detect()
            return self._wrap_detection(det)
        except Exception as e:
            print(f"[相机 API] 真实检测失败：{e}")
            return []

    def has_target(self) -> bool:
        return bool(self.detect_targets())
