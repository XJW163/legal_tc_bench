from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Checkpoint:
    """Small JSON checkpoint store with atomic writes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                # Preserve a broken file for diagnosis rather than silently overwriting it.
                broken = self.path.with_suffix(self.path.suffix + ".broken")
                try:
                    os.replace(self.path, broken)
                except OSError:
                    pass
                self.data = {}
        else:
            self.data = {}

        self.data.setdefault("version", 2)
        self.data.setdefault("completed", {})
        self.data.setdefault("updated", None)

    def done(self, key: Any) -> bool:
        return self.get(key) is not None

    def get(self, key: Any, default: Any = None) -> Any:
        return self.data.get("completed", {}).get(str(key), default)

    def mark(self, key: Any, meta: dict[str, Any] | None = None) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data.setdefault("completed", {})[str(key)] = {
            "time": now,
            "meta": meta or {},
        }
        self.data["updated"] = now
        self._save()

    def remove(self, key: Any) -> None:
        self.data.setdefault("completed", {}).pop(str(key), None)
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def _save(self) -> None:
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, self.path)
