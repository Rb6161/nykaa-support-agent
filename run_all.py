"""Run every demo in order, save each transcript to transcripts/, then fill the README results section.

    python run_all.py            (MOCK_LLM=true is the default; no API keys, no paid account)

Needs the dependencies from requirements.txt. The very first run downloads the free SentenceTransformers
model (all-MiniLM-L6-v2) once; afterwards everything runs locally."""
import config  # noqa: F401,E402

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
STEPS = [
    ("00_unit_tests", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ("01_dataset", [sys.executable, "dataset.py"]),
    ("02_escalation_score", [sys.executable, "orders.py"]),
    ("03_calibrate_threshold", [sys.executable, "calibrate.py"]),
    ("04_chunking_comparison", [sys.executable, "eval_chunking.py"]),
    ("05_grounded_generation", [sys.executable, "demo_rag.py"]),
    ("06_crew_tools_and_schema", [sys.executable, "demo_crew.py"]),
    ("07a_session_memory_multi_turn", [sys.executable, "demo_memory.py", "multi"]),
    ("07b_session_memory_fresh_conversation", [sys.executable, "demo_memory.py", "fresh"]),
    ("08_guardrails", [sys.executable, "demo_guardrails.py"]),
    ("09_evaluation_15_queries", [sys.executable, "evaluate.py"]),
    ("10_fastapi_websocket_logging", [sys.executable, "demo_api.py"]),
    ("11_autogen_review", [sys.executable, "demo_review.py"]),
    ("12_governance_budget_cache", [sys.executable, "demo_governance.py"]),
]

if __name__ == "__main__":
    (ROOT / "transcripts").mkdir(exist_ok=True)
    log = ROOT / "logs" / "requests.jsonl"
    log.parent.mkdir(exist_ok=True)
    log.write_text("")                                   # start the audit log fresh for this run
    env = {**os.environ, "MOCK_LLM": "true", "PYTHONIOENCODING": "utf-8"}
    failures = []
    for name, cmd in STEPS:
        t = time.time()
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        out = f"$ {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}\n\n{p.stdout}"
        if p.stderr.strip():
            out += f"\n--- stderr (warnings are expected, e.g. LangChain deprecation) ---\n{p.stderr[-3000:]}"
        out = out.replace(str(ROOT), "<repo>").replace(str(Path.home()), "~")   # keep local usernames/paths out of the repo
        (ROOT / "transcripts" / f"{name}.txt").write_text(out, encoding="utf-8")
        status = "ok" if p.returncode == 0 else "FAILED"
        print(f"[{status:>6}] {name}  ({time.time() - t:.1f}s)")
        if p.returncode:
            failures.append(name)
            print(p.stderr[-1500:])
    subprocess.run([sys.executable, "fill_readme.py"], cwd=ROOT, env=env)
    print("\nAll steps passed." if not failures else f"\nFailed steps: {failures}")
    sys.exit(1 if failures else 0)
