"""Tasks 15 + 16 demo: least autonomy, risk level, runtime budget cap, and response caching with
before/after evidence."""
import config  # noqa: F401,E402

import json
import time

import rag
from crew import build_crew, make_agent
from governance import MAX_REQUEST_TOKENS, RISK_CLASSIFICATION, enforce_budget
from mock_llm import get_llm
from pipeline import SupportPipeline
from tools import CheckOrderStatusTool, RagLookupTool

if __name__ == "__main__":
    results = {}

    print("##### APPLICATION LAYER: least autonomy (only the Lookup Agent may call check_order_status) #####")
    crew = build_crew("example question", "NYK-0001")
    wiring = {a.role: [t.name for t in a.tools] for a in crew.agents}
    print("actual tool wiring in the crew:", json.dumps(wiring))
    assert [r for r, t in wiring.items() if "check_order_status" in t] == ["Lookup Agent"]
    for role in ("Retrieval Agent", "Response Composer"):
        try:
            make_agent(role, "g", "b", [CheckOrderStatusTool()], get_llm())
            raise SystemExit(f"FAIL: {role} was allowed to hold check_order_status")
        except PermissionError as e:
            print(f"attempt to wire the tool to '{role}' BLOCKED -> {e}")
    print("Guard: governance.TOOL_ACL lists the roles allowed per tool, and crew.make_agent() calls authorize_tool() for "
          "every tool BEFORE an Agent object is created, so a mis-wired crew cannot even be constructed. The tool is "
          "also simply never passed to any other agent in build_crew().")
    results["tool_wiring"] = wiring

    print("\n##### APPLICATION LAYER: risk classification #####")
    print(f"Risk level: {RISK_CLASSIFICATION['level']}\n{RISK_CLASSIFICATION['justification']}")

    print("\n##### RUNTIME LAYER: per-request token/cost budget cap #####")
    print("cap =", MAX_REQUEST_TOKENS, "tokens")
    p = SupportPipeline()
    ok = p.ask("How long does a COD refund take?", "gov-1", "demo_governance")
    print(f"normal request  : refused={ok.refused}, estimated_tokens={ok.estimated_tokens}")
    big = p.ask("Please explain the refund policy in detail. " * 400, "gov-2", "demo_governance")
    print(f"oversized request: refused={big.refused}, estimated_tokens={big.estimated_tokens}, reason={big.refusal_reason}")
    print("answer:", big.answer)
    assert not ok.refused and big.refused and big.refusal_reason.startswith("budget_exceeded")
    print("budget detail:", enforce_budget("short"))
    results["budget"] = {"cap": MAX_REQUEST_TOKENS, "normal_tokens": ok.estimated_tokens, "oversized_tokens": big.estimated_tokens}

    print("\n##### RUNTIME LAYER: response cache (keyed by normalised query text) #####")
    rag.CACHE.clear()
    rag.COUNTERS.update(generation_calls=0, retrieval_calls=0)
    q1, q2 = "How long does a COD refund take?", "  how LONG does a cod refund take??  "
    print("BEFORE: counters =", rag.COUNTERS, "| cache =", rag.CACHE.stats())
    t = time.perf_counter(); r1 = rag.grounded_answer(q1); d1 = (time.perf_counter() - t) * 1000
    mid_counters, mid_stats = dict(rag.COUNTERS), rag.CACHE.stats()
    print(f"CALL 1 ({d1:.1f} ms): cache_hit={r1['cache_hit']} | counters = {mid_counters} | cache = {mid_stats}")
    t = time.perf_counter(); r2 = rag.grounded_answer(q2); d2 = (time.perf_counter() - t) * 1000
    print(f"CALL 2 ({d2:.1f} ms): cache_hit={r2['cache_hit']} | counters = {rag.COUNTERS} | cache = {rag.CACHE.stats()}")
    assert r2["cache_hit"] and rag.COUNTERS == mid_counters, "repeated query must not trigger new retrieval/generation"
    print(f"AFTER: the repeated (normalised-identical) query caused 0 extra generation calls and 0 extra retrievals; "
          f"{d1:.1f} ms -> {d2:.1f} ms")
    results["cache"] = {"first_ms": round(d1, 2), "second_ms": round(d2, 2), "counters_after_first": mid_counters,
                        "counters_after_second": dict(rag.COUNTERS), "stats": rag.CACHE.stats()}

    print("\n-- same effect end to end through the crew (tool-level cache hit) --")
    a = p.ask("What is the return window for beauty products?", "gov-3", "demo_governance")
    b = p.ask("What is the return window for beauty products?", "gov-4", "demo_governance")
    print(f"request 1: cache_hit={a.cache_hit}, latency={a.latency_ms} ms\nrequest 2: cache_hit={b.cache_hit}, latency={b.latency_ms} ms")
    assert not a.cache_hit and b.cache_hit
    (rag.ROOT / "governance_results.json").write_text(json.dumps(results, indent=2))
    print("\nOK: least autonomy enforced, risk classified, oversized request rejected, cache hit demonstrated.")
