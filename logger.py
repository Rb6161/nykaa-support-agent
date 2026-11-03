"""ELK-style structured logging: ONE JSON-Lines entry per request, with trace id and timings.

Every string value is passed through the SAME mask_pii() used for model input, so a phone number
or card-last-4 never reaches disk in the clear."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from guardrails import contains_raw_pii, mask_text

LOG_PATH = Path(__file__).parent / "logs" / "requests.jsonl"
_lock = threading.Lock()


def _mask_obj(obj):
    if isinstance(obj, str):
        return mask_text(obj)
    if isinstance(obj, dict):
        return {k: _mask_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_mask_obj(v) for v in obj]
    return obj


def log_request(entry: dict, path: Path = None) -> dict:
    path = path or LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **_mask_obj(entry)}
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


def scan_log_for_raw_pii(path: Path = None) -> list:
    """Returns the log lines that still contain raw phone/card digits (should be empty)."""
    path = path or LOG_PATH
    if not path.exists():
        return []
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        # trace ids / timestamps contain digits but never match the phone/card formats
        if contains_raw_pii(line):
            bad.append((i, line))
    return bad
