# app/vision/providers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from app.models.room import VisionOutput


class VisionProvider(ABC):
    @abstractmethod
    def analyze(self, image_base64: str) -> VisionOutput:
        """Return a structured VisionOutput from a base64 image."""
        raise NotImplementedError
