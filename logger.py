"""ELK-style structured logging: ONE JSON-Lines entry per request, with trace id and timings.

Every string value is passed through the SAME mask_pii() used for model input, so a phone number
or card-last-4 never reaches disk in the clear."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from guardrails import contains_raw_pii, mask_text

LOG_PATH = Path(__file__).parent / "logs" / "requests.jsonl"
ESCALATION_PATH = Path(__file__).parent / "logs" / "escalations.jsonl"   # hand-off queue read by backend support engineers
_lock = threading.Lock()


def new_trace_id() -> str:
    """12 hex chars, unique per request. Two positions are forced to be letters so an id can never contain a
    long digit run that the PII masker would mistake for a phone/card number and rewrite in the log."""
    h = list(uuid.uuid4().hex[:12])
    for i in (3, 7):
        h[i] = "abcdef"[int(h[i], 16) % 6]
    return "".join(h)


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


def log_escalation(trace_id: str, session_id: str, reason: str, masked_query: str, path: Path = None) -> dict:
    """Queue a case for the backend support engineers (human review). Same masking as every other log."""
    return log_request({"trace_id": trace_id, "session_id": session_id, "escalated_to": "backend_support_engineers",
                        "reason": reason, "query": masked_query, "status": "open"}, path or ESCALATION_PATH)
