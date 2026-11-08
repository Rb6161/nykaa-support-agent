# Nykaa Domain Support Agent (CrewAI + RAG + Autogen review + FastAPI)

**Track completed: Nykaa (E-commerce & Retail).**

One public repository containing the dataset, the RAG core, the CrewAI agent crew, the Autogen review stage, the governance controls and the FastAPI deployment. Everything runs under `MOCK_LLM` (default) with **zero API keys, zero paid accounts and no outbound network calls at run time**.

---

## 1. Part 1 / Task 1 dataset-design choices (for deterministic reproduction)

`dataset.py` is a seeded generator (`random.Random(SEED)`). Running it twice gives identical records.

| choice | value |
|---|---|
| **seed** | `SEED = 2026` |
| records | 240 (`record_id` = `NYK-0001` ... `NYK-0240`) |
| **category weights** | Beauty 0.30, Apparel 0.25, Footwear 0.15, Home 0.15, Electronics 0.15 |
| **status weights** | Delivered 0.45, Shipped 0.20, Placed 0.15, Returned 0.12, Refunded 0.08 |
| **amount range (`order_value_inr`)** | uniform per category: Beauty 199-3,499; Apparel 499-5,999; Footwear 799-6,999; Home 399-7,999; Electronics 999-14,999 (overall 199-14,999 INR; generated: 230-14,611) |
| `days_since_created` | integer 0-30, status-aware (Placed 0-6, Shipped 1-14, Delivered 3-30, Returned 8-30, Refunded 10-30) |
| `delayed_shipment` | Bernoulli by status: Placed 0.12, Shipped 0.30, Delivered 0.18, Returned 0.20, Refunded 0.20 |

**Price-range reasoning (one sentence):** Nykaa sells mostly beauty and fashion items priced from a few hundred to a few thousand rupees, with occasional higher-ticket electronics such as hair tools and grooming devices.

**Reported by `python dataset.py` (seed 2026):** Beauty 68, Apparel 73, Footwear 35, Electronics 35, Home 29 (every category >= 3); Delivered 109, Shipped 47, Placed 39, Returned 31, Refunded 14 (every status >= 1); **delayed_shipment = True for 15.83 %** (required band 10-30 %). No record was hand-edited.

Fabricated data only. Customer name and delivery address are free text with no reliable pattern for a keyless masker, so they are **out of scope for masking**. The fixed-format PII fields **phone number** and **payment-card last-4 digits** are masked (Task 10).

---

## 2. Setup and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # Python 3.10-3.12 recommended

python run_all.py                      # runs every demo, saves transcripts/, fills section 7 of this README
uvicorn app:app --port 8000            # start the API (after run_all.py has calibrated the threshold)
```

* The first run downloads the free local SentenceTransformers model `all-MiniLM-L6-v2` once (from Hugging Face). After that, `rag.py` loads it with `local_files_only=True`, so later runs make **no network access at all** (you can also set `HF_HUB_OFFLINE=1`).
* **Telemetry confirmation:** `config.py` is imported first by every entry point and sets `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_DISABLE_TRACKING=true` and `ANONYMIZED_TELEMETRY=False` *before* CrewAI or Chroma are imported, so `crew.kickoff()` makes no outbound call. If you run things by hand, you can also export them yourself:
  * macOS/Linux: `export CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true`
  * Windows PowerShell: `$env:CREWAI_DISABLE_TELEMETRY="true"; $env:OTEL_SDK_DISABLED="true"`
* `MOCK_LLM=true` is the default. Setting `MOCK_LLM=false` plus `LLM_MODEL=<model>` is wired only as an optional path; **every acceptance criterion is demonstrated under the mock**.
* Unit tests that need no heavy dependencies: `python -m pytest -q tests`.

---

## 3. Repository map

| file | what it does |
|---|---|
| `dataset.py` | seeded generator, `ORDERS` (240 records), validation report |
| `policy_docs/*.md` | 14 knowledge-base documents (up to 5 sentences each) covering all 12 required topics, plus order-status meanings and invoices/gift packaging |
| `rag.py` | fixed-size-with-overlap and sentence chunking, SentenceTransformers embeddings, **two separate ChromaDB collections** (`upsert`), calibrated threshold, grounded generation, cached entry point |
| `calibrate.py` | measures top-1 similarities and sets the "I don't know" threshold |
| `eval_chunking.py` | document-level precision/recall for both collections with per-query arithmetic + recommendation |
| `orders.py` | `check_order_status`, designed escalation score and threshold |
| `mock_llm.py` | deterministic `MockCrewLLM(BaseLLM)` for CrewAI (no keys, no network) |
| `tools.py` | CrewAI tools `rag_lookup` and `check_order_status` (with a call log for proof) |
| `crew.py` | 3-agent crew (Retrieval, Lookup, Response Composer), run with `.kickoff()` |
| `memory.py` | LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` |
| `schemas.py` | all Pydantic models (`CrewAnswer`, `Verdict`, API models) |
| `guardrails.py` | PII masking, prompt-injection detection, groundedness check |
| `review.py`, `review_rules.py` | Autogen `RoundRobinGroupChat` review team (Policy-Compliance-Reviewer + Final-Editor) |
| `governance.py` | least-autonomy ACL, risk classification, per-request budget cap |
| `cache.py` | in-memory response cache keyed by normalised query |
| `logger.py` | JSON-Lines structured logging with PII masking |
| `pipeline.py` | wires everything: guardrails -> budget -> memory -> crew -> groundedness -> review -> log |
| `app.py` | FastAPI: `POST /ask`, `POST /add-document`, `GET /health`, `WS /ws/chat` |
| `judge.py`, `evaluate.py`, `queries.py` | LLM-as-judge (mock) and the 15-query evaluation |
| `demo_*.py`, `run_all.py` | demonstrations that write `transcripts/*.txt` |

---

## 4. How each task is satisfied

**Part 1**
1. *Dataset* - `dataset.py` (section 1).
2. *Knowledge base* - `policy_docs/` (14 docs, each topic covered once or more).
3. *Two chunking strategies* - fixed windows of 35 words with 10-word overlap, and one chunk per sentence; embedded with `all-MiniLM-L6-v2` and upserted into **separate** collections `nykaa_fixed_chunks` and `nykaa_sentence_chunks` (cosine space).
4. *Grounded generation* - retrieves top-4 chunks and answers **only** from them; if the top-1 cosine similarity is below the calibrated threshold it returns the "I don't know" fallback. The threshold is measured by `calibrate.py` (>= 3 in-scope and >= 2 out-of-scope queries) and set midway between the observed clusters; no 0.5/0.6/0.7 presets. Measured values are in section 7. `demo_rag.py` runs 25 in-scope queries and 1 out-of-scope query that triggers the fallback.
5. *Chunking comparison* - `eval_chunking.py`: chunks are mapped to parent documents and deduplicated before scoring; per-query arithmetic is printed for both collections, followed by the recommendation (section 7).

**Part 2**
6. *Order tool + escalation score* - `check_order_status(record_id)` returns `status`, `order_value_inr` and `escalation_score` in [0, 1]:

   `escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)`

   A delayed shipment is the strongest signal (0.6); the normalised age of the order is a secondary signal (0.4): the longer an order has been open, the more urgent it is (I used age as the "recency" signal). **Escalation threshold = 0.3733**, the **80th percentile of the escalation score over my 240 generated orders** (the 80th percentile of `days_since_created` is 23 days). It flags 52 of 240 orders (21.7 %): all 38 delayed orders plus the 14 oldest non-delayed ones. It is computed from the data in `orders.py`, not hard-coded. (`python orders.py` prints it.)
7. *Crew* - Retrieval Agent (RAG tool), Lookup Agent (order tool), Response Composer, run via `crew.kickoff()`; `demo_crew.py` shows the RAG tool invoked on a policy query and the order tool invoked on order queries (via `tools.TOOL_CALL_LOG`).
8. *Session memory* - `memory.py`; `demo_memory.py` shows an order id given in turn 1 being reused in later turns, then a fresh session where the same follow-up correctly asks for the order id. (`RunnableWithMessageHistory` raises a `LangChainDeprecationWarning`; this is expected and not silenced.)
9. *Structured output* - `CrewAnswer` is set as `output_pydantic` on the composer task and every response is re-validated in code (`crew.run_crew`).
10. *Guardrails* - input: phone/card-last-4 masking and prompt-injection detection; output: groundedness check that refuses when the retrieved context does not support the answer. `demo_guardrails.py` fires each one on a deliberate test case.

**Part 3**
11. *FastAPI* - `POST /ask`, `POST /add-document` (Pydantic request/response models; `/ask` returns 413 for a budget-exceeding request), `GET /health`, and `WS /ws/chat`, which catches `WebSocketDisconnect` and keeps serving other clients. `demo_api.py` disconnects a client mid-conversation and then shows the server still answering.
12. *Structured logging* - `logs/requests.jsonl`, one JSON-Lines entry per request with `trace_id`, endpoint, timings and flags. The same `mask_pii()` used for the model input is applied to every logged string, so no raw phone/card value reaches disk (`demo_api.py` asserts this by scanning the file).
13. *Evaluation* - `evaluate.py`: 15 queries (>= 1 per KB topic: 12; plus an out-of-scope query, a prompt-injection query and a PII query), LLM-as-judge prompt run by a deterministic mock judge, four scores per query and the four averages.

**Part 4**
14. *Autogen review* - `review.py`: `RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])`; the Final-Editor uses `output_content_type=Verdict` (`approved: bool`, `final_answer: str`, `reason: str`). The input is the Composer's draft plus the retrieved context. `demo_review.py` shows one draft **approved unchanged** and one draft with a **deliberately injected ungrounded claim that is revised**.
15. *Four-layer governance* - Data (fabricated records, PII masked before the model and the logs), Model (deterministic `MOCK_LLM`, no external calls), Application (least autonomy, Pydantic schemas, independent review), Runtime (token/cost budget cap, cache, JSONL audit log).
    * **Least autonomy:** only the Lookup Agent may hold `check_order_status`. `governance.TOOL_ACL` lists the allowed roles per tool and `crew.make_agent()` calls `authorize_tool()` for every tool *before* an `Agent` exists, so wiring the tool to any other agent raises `PermissionError`; `build_crew()` also simply never passes it to another agent. `demo_governance.py` prints the real wiring and shows the blocked attempts.
    * **Risk level: Medium** (customer-support tickets). It is not Low because it reads customer order records and receives phone numbers and card last-4 digits. It is not High because it handles no medical data, makes no hiring or credit decisions, cannot move money (it only quotes refund policy), and has read-only access to fabricated data. Answers are grounded, reviewed by a second agent team, and refused or escalated when confidence is low.
    * **Runtime budget cap:** `governance.enforce_budget()` estimates tokens (`chars / 4` plus a fixed 700-token prompt overhead) against a 1,000-token cap; an oversized request is **rejected** (HTTP 413 from the API), never silently run.
16. *Response caching* - `rag.grounded_answer()` is cached by normalised query text. `demo_governance.py` prints before/after call counters and timings: the repeated query causes **0** additional retrieval or generation calls.

---

## 5. Two silent CrewAI pitfalls handled in `mock_llm.py`

1. CrewAI's ReAct system prompt itself contains the text `Observation: the result of the action`. The mock never searches the system prompt, only the conversation turns after it, and ignores that placeholder text, so it cannot return template text as a tool result. (`tests/test_mock_llm.py` checks this.)
2. Tool arguments are chosen from each tool's own **declared argument schema** (`Tool Arguments: {...}`), never by testing whether `"lookup"` is a substring of the tool name, so a tool named `rag_lookup` is not mistaken for the order-lookup tool. (Also tested.)

---

## 6. Honest notes

* The "LLM" calls (grounded generation, judge, CrewAI agents, Autogen agents) are deterministic and rule-based under `MOCK_LLM`. The CrewAI crew uses a real `crewai.Crew`/`Agent`/`Task` stack with a `BaseLLM` subclass, and the review stage uses real Autogen agents and a real `RoundRobinGroupChat`; only the model behind the agents is a mock (the Autogen agents use a replay model client whose replies come from `review_rules.py`).
* Mock grounded generation is extractive: it returns the two retrieved sentences most similar to the question.
* Customer name and delivery address are not masked (out of scope per the brief).
* The all-1.00 evaluation averages come from a deterministic **mock judge** with simple rules (right source document, sentence support in the context, key terms present, no raw PII, correct refusals). They show the pipeline behaves as designed on the 15 queries; they are not a measure of real language quality.
* Because mock generation is extractive, the second selected sentence is sometimes only loosely relevant (for example, query 3 in `transcripts/09_evaluation_15_queries.txt` also returns a pickup-attempts sentence). It is still grounded in the knowledge base, so it passes the groundedness checks, but a real LLM would answer more tightly.

---

## 7. Measured results

<!-- RESULTS:START -->
### Measured similarity threshold (Task 4)

**fixed collection** - in-scope top-1 similarities: 0.712, 0.663, 0.658, 0.700, 0.606 (min 0.606); out-of-scope: 0.056, 0.130, 0.075, 0.099 (max 0.130). **Chosen threshold = 0.368** (midpoint of the two clusters).

**sentence collection** - in-scope top-1 similarities: 0.775, 0.734, 0.714, 0.709, 0.629 (min 0.629); out-of-scope: 0.143, 0.101, 0.099, 0.164 (max 0.164). **Chosen threshold = 0.3968** (midpoint of the two clusters).

### Chunking comparison on 25 queries (Task 5)

| strategy | chunks | mean precision | mean recall | F1 |
|---|---|---|---|---|
| fixed | 53 | 0.503 | 1.000 | 0.670 |
| sentence | 84 | 0.660 | 1.000 | 0.795 |

I recommend deploying the sentence-based chunking. It scored mean precision 0.660 and mean recall 1.000 (F1 0.795) over the 25 queries, against precision 0.503, recall 1.000 (F1 0.670) for the fixed strategy. It indexes 84 chunks versus 53.

Per-query arithmetic: `transcripts/04_chunking_comparison.txt`.

### Evaluation, 15 queries under MOCK_LLM (Task 13)

| metric | average |
|---|---|
| accuracy | 1.000 |
| grounding | 1.000 |
| completeness | 0.933 |
| safety | 1.000 |

Per-query scores: `transcripts/09_evaluation_15_queries.txt`.

### Cache evidence (Task 16)

First call 37.72 ms (generation calls 1); repeated query 0.39 ms with generation calls still 1.

<!-- RESULTS:END -->

All transcripts proving each task are in `transcripts/`. Everything above runs under `MOCK_LLM` with no API key.
