"""CONCURRENCY test against a REAL uvicorn server: many simultaneous sessions must not interfere.

    python concurrency_test.py            # 12 HTTP sessions + 4 WebSocket clients
    python concurrency_test.py 1 0        # 1 session, no WebSocket (baseline)
    python concurrency_test.py 3 1        # 3 HTTP sessions + 1 WebSocket client
    python concurrency_test.py 12 4 --debug   # also dump all server thread stacks (use when something hangs)

Checks: no failed requests; every session sees ONLY its own order id (no memory leakage); trace ids are unique;
the log has a line per request; no raw PII in the log. Needs `python run_all.py` to have run once."""
import config  # noqa: F401,E402

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from websockets.sync.client import connect

from dataset import ORDERS

ROOT = Path(__file__).parent
PORT = int(os.environ.get("TEST_PORT", "8766"))
BASE, WS_URL, LOG = f"http://127.0.0.1:{PORT}", f"ws://127.0.0.1:{PORT}/ws/chat", ROOT / "logs" / "requests.jsonl"
ARGS = [x for x in sys.argv[1:] if not x.startswith("--")]
N = int(ARGS[0]) if len(ARGS) > 0 else 12          # simultaneous HTTP sessions
N_WS = int(ARGS[1]) if len(ARGS) > 1 else 4        # simultaneous WebSocket clients
DEBUG = "--debug" in sys.argv                      # make the server dump all thread stacks every 60 s
TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "240"))
PHONE = "9012345678"

results, lock, trace_ids, errors = [], threading.Lock(), [], []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def http_session(i):
    """Two turns in one session: the order id given in turn 1 must come back in turn 2, and be MY order."""
    oid = ORDERS[i * 7 % len(ORDERS)]["record_id"]
    sid = f"conc-{i}"
    try:
        with httpx.Client(base_url=BASE, timeout=TIMEOUT) as c:
            a = c.post("/ask", json={"query": f"My order id is {oid}. My phone is {PHONE}. Please check it.", "session_id": sid})
            b = c.post("/ask", json={"query": "What is the status of my order?", "session_id": sid})
            p = c.post("/ask", json={"query": "How long does a COD refund take?", "session_id": f"conc-policy-{i}"})
        ja, jb, jp = a.json(), b.json(), p.json()
        with lock:
            trace_ids.extend([ja["trace_id"], jb["trace_id"], jp["trace_id"]])
        ok = (a.status_code == b.status_code == p.status_code == 200
              and ja["order"] and ja["order"]["record_id"] == oid
              and jb["order"] and jb["order"]["record_id"] == oid
              and "02_cod_refund" in jp["sources"])
        if not ok:
            errors.append(f"session {i}: expected {oid}, got turn1={ja.get('order')} turn2={jb.get('order')} policy_sources={jp.get('sources')}")
        return ok
    except Exception as e:
        errors.append(f"session {i}: {e!r}")
        return False


def ws_client(i):
    oid = ORDERS[(i * 11 + 3) % len(ORDERS)]["record_id"]
    try:
        with connect(WS_URL, open_timeout=60, close_timeout=5) as ws:
            ws.send(json.dumps({"query": f"My order is {oid}, can you check it?"}))
            m1 = json.loads(ws.recv(timeout=TIMEOUT))
            ws.send(json.dumps({"query": "What is the status of my order?"}))
            m2 = json.loads(ws.recv(timeout=TIMEOUT))
        with lock:
            trace_ids.extend([m1["trace_id"], m2["trace_id"]])
        ok = bool(m2["order"]) and m2["order"]["record_id"] == oid
        if not ok:
            errors.append(f"ws {i}: expected {oid}, got {m2.get('order')}")
        return ok
    except Exception as e:
        errors.append(f"ws {i}: {e!r}")
        return False


def main():
    if not (ROOT / "calibration.json").exists():
        print("Run `python run_all.py` first.")
        return 2
    before = len(LOG.read_text(encoding="utf-8").splitlines()) if LOG.exists() else 0
    server_log_path = Path(tempfile.gettempdir()) / "nykaa_concurrency_server_output.txt"
    server_log = open(server_log_path, "w", encoding="utf-8")        # a FILE: an unread pipe fills up and freezes the server
    env = {**os.environ, "MOCK_LLM": "true", "PYTHONIOENCODING": "utf-8"}
    if DEBUG:
        env["DEBUG_STACKS"] = "1"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--log-level", "warning"],
                              cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(240):
            if server.poll() is not None:
                print("server exited early"); return 1
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            print("server never became healthy"); return 1
        print(f"server up; firing {N} HTTP sessions (3 requests each) and {N_WS} WebSocket clients at the same time (timeout {TIMEOUT}s) ...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N + N_WS + 1) as pool:
            http_futs = [pool.submit(http_session, i) for i in range(N)]
            ws_futs = [pool.submit(ws_client, i) for i in range(N_WS)]
            http_ok = [f.result() for f in http_futs]
            ws_ok = [f.result() for f in ws_futs]
        print(f"finished in {time.time() - t0:.1f}s")
        check(f"all {N} HTTP sessions returned correct, session-isolated answers", all(http_ok), f"{sum(http_ok)}/{N}")
        check(f"all {N_WS} concurrent WebSocket clients got correct, session-isolated answers", all(ws_ok), f"{sum(ws_ok)}/{N_WS}")
        check("trace ids are unique across all responses", len(set(trace_ids)) == len(trace_ids), f"{len(trace_ids)} responses")
        time.sleep(1)
        lines = LOG.read_text(encoding="utf-8").splitlines()[before:]
        logged = {json.loads(l).get("trace_id") for l in lines}
        check("every response has a matching log entry", all(t in logged for t in trace_ids), f"{len(lines)} new log lines")
        check("the raw phone number never reached the log", PHONE not in LOG.read_text(encoding="utf-8"))
        check("server is still healthy after the burst", httpx.get(f"{BASE}/health", timeout=10).status_code == 200)
        for e in errors[:10]:
            print("   problem:", e)
        if errors:
            server_log.flush()
            tail = server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
            print(f"\n--- last lines of the server's own output ({server_log_path}) ---")
            print("\n".join(tail))
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
