"""Session memory with LangChain: InMemoryChatMessageHistory + RunnableWithMessageHistory.

Note: RunnableWithMessageHistory emits a LangChainDeprecationWarning that points to LangGraph's
persistence layer. That is expected and intentionally not silenced; the class still works here.
Memory is in-process only (it does not survive a restart)."""
from __future__ import annotations

from typing import Callable, Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


class ConversationMemory:
    def __init__(self, turn_fn: Callable[[dict], object]):
        """turn_fn receives {"input": <user text>, "history": [BaseMessage, ...]} and returns an AIMessage."""
        self._histories: Dict[str, InMemoryChatMessageHistory] = {}
        self._chain = RunnableWithMessageHistory(
            RunnableLambda(turn_fn), self._get_history,
            input_messages_key="input", history_messages_key="history",
        )

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        return self._histories.setdefault(session_id, InMemoryChatMessageHistory())

    def invoke(self, text: str, session_id: str):
        return self._chain.invoke({"input": text}, config={"configurable": {"session_id": session_id}})

    def history(self, session_id: str) -> List[str]:
        h = self._histories.get(session_id)
        return [f"{m.type}: {m.content}" for m in h.messages] if h else []

    def sessions(self) -> List[str]:
        return list(self._histories)
