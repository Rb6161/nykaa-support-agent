"""Import this module FIRST in every entry point.

It forces the zero-network / zero-key defaults required by the brief:
  * MOCK_LLM defaults to "true" (deterministic mock models, no API keys)
  * CrewAI telemetry (outbound network call on crew.kickoff()) is disabled
  * Chroma anonymised telemetry is disabled
"""
import os

os.environ.setdefault("MOCK_LLM", "true")
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def mock_llm_enabled() -> bool:
    return os.environ.get("MOCK_LLM", "true").lower() != "false"
