"""FastAPI deployment:  uvicorn app:app --port 8000

  POST /ask           -> AskRequest  -> AskResponse        (one JSONL log line per request)
  POST /add-document  -> AddDocumentRequest -> AddDocumentResponse
  GET  /health
  WS   /ws/chat       real-time multi-turn chat; a client disconnecting mid-conversation is caught
                      (WebSocketDisconnect) and never takes the server down for other clients."""
from __future__ import annotations

import config  # noqa: F401

import asyncio
import faulthandler
import json
import os
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

import rag
from logger import log_request
from pipeline import SupportPipeline
from schemas import AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse



@asynccontextmanager
async def lifespan(_app: FastAPI):
    # warm the embedding model + both Chroma collections, and fail fast if calibration is missing
    await asyncio.to_thread(rag.get_engine)
    rag.threshold_for(rag.recommended_strategy())
    yield


if os.environ.get("DEBUG_STACKS"):
    # debugging aid: print the stack of EVERY thread to stderr every 60 s, so a hang shows exactly where it is stuck
    faulthandler.dump_traceback_later(60, repeat=True)

app = FastAPI(title="Nykaa Domain Support Agent", version="1.0", lifespan=lifespan)
pipeline = SupportPipeline()


@app.middleware("http")
async def log_remaining_requests(request: Request, call_next):
    """/ask and /add-document write their own (richer) entry. Everything else (e.g. /health, 404s and
    request-validation failures) gets a minimal entry here, so EVERY request yields one JSONL line."""
    t0 = time.perf_counter()
    response = await call_next(request)
    own_entry = request.url.path in ("/ask", "/add-document") and response.status_code != 422
    if not own_entry:
        log_request({"trace_id": uuid.uuid4().hex[:12], "endpoint": request.url.path, "method": request.method,
                     "status_code": response.status_code, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
    return response


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": config.mock_llm_enabled(), "strategy": rag.recommended_strategy()}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    resp = await asyncio.to_thread(pipeline.ask, req.query, req.session_id, "/ask")
    if (resp.refusal_reason or "").startswith("budget_exceeded"):
        raise HTTPException(status_code=413, detail={"error": "request exceeds token budget", "trace_id": resp.trace_id})
    return resp


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(req: AddDocumentRequest):
    trace_id, t0 = uuid.uuid4().hex[:12], time.perf_counter()
    info = await asyncio.to_thread(rag.get_engine().add_document, req.title, req.text)
    log_request({"trace_id": trace_id, "endpoint": "/add-document", "query": f"{req.title}: {req.text}",
                 "latency_ms": round((time.perf_counter() - t0) * 1000, 2), **info})
    return AddDocumentResponse(trace_id=trace_id, **info)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex[:8]
    try:
        while True:
            raw = await ws.receive_text()
            try:
                query = json.loads(raw).get("query", "")
            except (ValueError, AttributeError):
                query = raw
            resp = await asyncio.to_thread(pipeline.ask, query, session_id, "/ws/chat")
            await ws.send_json(resp.model_dump(mode="json"))
    except WebSocketDisconnect:
        log_request({"trace_id": uuid.uuid4().hex[:12], "endpoint": "/ws/chat", "session_id": session_id,
                     "event": "client_disconnected"})
    except Exception as e:  # never let one client's failure kill the server
        log_request({"trace_id": uuid.uuid4().hex[:12], "endpoint": "/ws/chat", "session_id": session_id,
                     "event": "error", "error": repr(e)})
        try:
            await ws.close(code=1011)
        except Exception:
            pass
