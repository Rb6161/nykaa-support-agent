"""Data-driven test runner: reads test_cases.json (edit/add your own cases) and checks each one.

    python run_cases.py                      # runs every case in test_cases.json
    python run_cases.py my_other_cases.json  # use another file

Supported "expect" keys: status, status_not, grounded, refused, refusal_prefix, pii_masked, order_id,
source_contains, answer_contains, answer_not_contains, cache_hit, raw_absent (text that must NOT be in the answer or the log).
Runs in-process (FastAPI TestClient). Needs `python run_all.py` to have run once."""
import config  # noqa: F401,E402

import json
import os
import socket
import sys
from pathlib import Path

# ---- OFFLINE GUARD: prove the run needs no network and no API key --------------------------------
assert os.environ.get("MOCK_LLM", "true").lower() == "true", "MOCK_LLM must be true for the offline sanity run"
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY"):
    os.environ.pop(k, None)
_real_connect = socket.socket.connect
def _no_internet(self, addr, *a, **k):
    host = addr[0] if isinstance(addr, tuple) else str(addr)
    if host not in ("127.0.0.1", "::1", "localhost", "testserver") and not str(host).startswith("/"):
        raise OSError(f"OFFLINE GUARD: blocked outbound connection to {host}")
    return _real_connect(self, addr, *a, **k)
socket.socket.connect = _no_internet
print("offline guard ON: MOCK_LLM=true, API keys removed, outbound network blocked\n")

from fastapi.testclient import TestClient

from app import app
from logger import LOG_PATH

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "test_cases.json"
cases = json.loads(path.read_text(encoding="utf-8"))
fails = 0

with TestClient(app) as client:
    for c in cases:
        body = {"query": c["query"]}
        if c.get("session"):
            body["session_id"] = c["session"]
        r = client.post("/ask", json=body)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        e, problems = c.get("expect", {}), []
        if "status" in e and r.status_code != e["status"]: problems.append(f"status {r.status_code} != {e['status']}")
        if "status_not" in e and r.status_code == e["status_not"]: problems.append(f"status was {r.status_code}")
        if r.status_code == 200:
            ans = j.get("answer", "")
            for key in ("grounded", "refused"):
                if key in e and j.get(key) != e[key]: problems.append(f"{key}={j.get(key)} (wanted {e[key]})")
            if "refusal_prefix" in e and not str(j.get("refusal_reason")).startswith(e["refusal_prefix"]):
                problems.append(f"refusal_reason={j.get('refusal_reason')}")
            if "pii_masked" in e and j.get("pii_masked") != e["pii_masked"]: problems.append(f"pii_masked={j.get('pii_masked')}")
            if "order_id" in e and (j.get("order") or {}).get("record_id") != e["order_id"]: problems.append(f"order={j.get('order')}")
            if "cache_hit" in e and j.get("cache_hit") != e["cache_hit"]: problems.append(f"cache_hit={j.get('cache_hit')}")
            if "source_contains" in e and not any(e["source_contains"] in s for s in j.get("sources", [])): problems.append(f"sources={j.get('sources')}")
            if "answer_contains" in e and e["answer_contains"] not in ans: problems.append(f"answer lacks {e['answer_contains']!r}")
            if "answer_not_contains" in e and e["answer_not_contains"] in ans: problems.append(f"answer has {e['answer_not_contains']!r}")
            if "raw_absent" in e and e["raw_absent"] in ans: problems.append("raw value leaked into answer")
        if "raw_absent" in e and LOG_PATH.exists() and e["raw_absent"] in Path(LOG_PATH).read_text(encoding="utf-8"):
            problems.append("raw value leaked into the log")
        print(f"[{'FAIL' if problems else 'PASS'}] {c['name']}" + (f"  -> {'; '.join(problems)}" if problems else ""))
        fails += bool(problems)

print(f"\n{len(cases) - fails}/{len(cases)} cases passed")
sys.exit(1 if fails else 0)
