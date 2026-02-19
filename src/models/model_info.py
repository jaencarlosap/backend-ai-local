from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.models.enums import ModelState, TaskType

if TYPE_CHECKING:
    from src.inference.base import BaseInferenceEngine


@dataclass
class ModelInfo:
    model_id: str
    task_type: TaskType
    state: ModelState = ModelState.ON_DISK
    vram_mb: float = 0.0
    last_used: float = field(default_factory=time.time)
    disk_path: Optional[str] = None
    engine_instance: Optional[BaseInferenceEngine] = None

    def touch(self):
        self.last_used = time.time()
