import json
import threading
from pathlib import Path

import pytest

from legal_tc_bench.io_utils import atomic_write_json, exclusive_process_lock


def test_atomic_write_json_has_no_shared_tmp_collision(tmp_path: Path):
    path = tmp_path / "state.json"
    errors = []

    def writer(worker: int):
        try:
            for index in range(50):
                atomic_write_json(path, {"worker": worker, "index": index})
        except Exception as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    value = json.loads(path.read_text(encoding="utf-8"))
    assert set(value) == {"worker", "index"}
    assert not list(tmp_path.glob("*.tmp"))


def test_exclusive_process_lock_rejects_duplicate(tmp_path: Path):
    lock = tmp_path / ".repair.lock"
    with exclusive_process_lock(lock, label="first"):
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_process_lock(lock, label="second"):
                pass
