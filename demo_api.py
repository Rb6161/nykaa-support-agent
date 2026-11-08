"""Task 11/12 demo: exercise the FastAPI app in-process (TestClient): HTTP endpoints, a WebSocket chat, and a
client disconnecting mid-conversation without crashing the server; then verify the JSONL log."""
import config  # noqa: F401,E402

import json

from fastapi.testclient import TestClient

import rag
from app import app
from logger import LOG_PATH, scan_log_for_raw_pii

if __name__ == "__main__":
    LOG_PATH.parent.mkdir(exist_ok=True)
    before = len(LOG_PATH.read_text().splitlines()) if LOG_PATH.exists() else 0
    with TestClient(app) as client:
        print("GET /health ->", client.get("/health").json())

        r = client.post("/ask", json={"query": "How long does a COD refund take?", "session_id": "api-1"})
        print("\nPOST /ask ->", r.status_code, json.dumps({k: r.json()[k] for k in ("trace_id", "answer", "sources", "grounded", "cache_hit")}, indent=2))
        assert r.status_code == 200

        r = client.post("/ask", json={"query": "My phone is 9876543210. What is the status of order NYK-0003?", "session_id": "api-2"})
        print("\nPOST /ask (with phone number) ->", r.status_code, "pii_masked =", r.json()["pii_masked"], "| answer:", r.json()["answer"])

        r = client.post("/ask", json={"query": "word " * 5000})
        print("\nPOST /ask (oversized) ->", r.status_code, r.json())
        assert r.status_code == 413

        r = client.post("/ask", json={"nope": 1})
        print("POST /ask (invalid body) ->", r.status_code, "(Pydantic validation)")
        assert r.status_code == 422

        r = client.post("/add-document", json={"title": "Gift card policy",
                        "text": "Gift cards are valid for 12 months from the date of purchase. They can be used on any Nykaa order but cannot be exchanged for cash."})
        print("\nPOST /add-document ->", r.status_code, r.json())
        r = client.post("/ask", json={"query": "How long are gift cards valid?"})
        print("ask about the new document ->", r.json()["answer"], "| sources:", r.json()["sources"])

        print("\n--- WebSocket /ws/chat: client 1 chats, then disconnects mid-conversation ---")
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"query": "My order is NYK-0003. Can you check it?"}))
            print("ws reply 1:", ws.receive_json()["answer"])
            ws.send_text(json.dumps({"query": "What is the status of my order?"}))
            print("ws reply 2 (memory):", ws.receive_json()["answer"])
            ws.send_text(json.dumps({"query": "How long does delivery take in metro cities?"}))
        print("client 1 closed the socket before reading the 3rd reply (abrupt disconnect)")

        print("--- server still alive: health check + a second client ---")
        print("GET /health ->", client.get("/health").json())
        with client.websocket_connect("/ws/chat") as ws2:
            ws2.send_text("What does the Delivered status mean?")
            print("client 2 reply:", ws2.receive_json()["answer"])

    added = rag.POLICY_DIR / "added_gift_card_policy.md"      # demo artefact: remove so reruns start from the original KB
    if added.exists():
        added.unlink()

    lines = LOG_PATH.read_text().splitlines()[before:]
    print(f"\n--- JSONL log: {len(lines)} new entries (one per request), each with trace_id + timing ---")
    for l in lines[:3]:
        print(l)
    assert all("trace_id" in json.loads(l) for l in lines)
    assert not scan_log_for_raw_pii(), "raw PII found in log"
    assert "9876543210" not in LOG_PATH.read_text()
    print("\nOK: no raw phone/card digits anywhere in the log file; server survived the disconnect.")
