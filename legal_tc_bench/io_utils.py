from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace *path* using a process-unique temporary file.

    The temporary file is created in the destination directory so os.replace remains
    atomic. A unique name avoids the fixed ``file.json.tmp`` collision that occurs
    when two processes happen to write the same destination at the same time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@contextmanager
def exclusive_process_lock(lock_path: Path, *, label: str) -> Iterator[TextIO]:
    """Hold a non-blocking process lock for a whole output task.

    On macOS/Linux this uses ``fcntl.flock`` and is automatically released if the
    process exits. The lock file may remain on disk; its presence alone does not mean
    it is active. A second process targeting the same output directory fails early
    with a useful error instead of corrupting checkpoints or racing on temporary files.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            import fcntl  # macOS/Linux

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Another process is already running {label} for this output. "
                f"Lock: {lock_path}. Owner: {owner}. "
                "Do not start two workers for the same experiment/output directory."
            ) from exc
        except ImportError:
            # Conservative fallback for platforms without fcntl. This repository is
            # primarily run on macOS/Linux; the unique atomic temp names still protect
            # individual writes on other systems.
            acquired = True

        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "label": label,
                },
                ensure_ascii=False,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield handle
    finally:
        if acquired:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        handle.close()
