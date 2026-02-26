"""
Router agent for mode detection.

Determines whether to use full-book RAG or selected-text mode.
"""
from typing import Literal
from app.models.chat import ChatRequest


class RouterAgent:
    """Router agent for chat mode detection."""

    def detect_mode(self, request: ChatRequest) -> Literal["full_book", "selected_text"]:
        """
        Detect which chat mode to use based on request.

        Args:
            request: Chat request

        Returns:
            Mode: "full_book" or "selected_text"
        """
        # Explicit mode field takes precedence
        if request.mode:
            return request.mode

        # Auto-detect based on selected_text presence
        if request.selected_text and request.selected_text.strip():
            return "selected_text"

        return "full_book"


# Global router agent instance
router_agent = RouterAgent()
