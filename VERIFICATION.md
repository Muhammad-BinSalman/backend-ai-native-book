"""
Constitution Compliance Verification

Verifies all constitution principles are followed.
"""
from app.config.settings import settings
from app.services.cohere_service import cohere_service

def verify_compliance() -> dict:
    """Verify all constitution constraints."""
    compliance = {
        "strict_grounding": False,
        "cohere_first": False,
        "free_tier": False,
        "openai_agents_sdk": False,
        "type_safety": False,
    }

    # 1. Strict Grounding - Check system prompt
    from app.agents.rag_agent import RAGAgent
    if "ONLY the retrieved book passages" in RAGAgent.SYSTEM_PROMPT:
        compliance["strict_grounding"] = True

    # 2. Cohere-First - Check base URL
    base_url = cohere_service.verify_base_url()
    if "api.cohere.ai" in base_url and "compatibility" in base_url:
        compliance["cohere_first"] = True

    # 3. Free-Tier - Check storage monitoring
    from scripts.ingest_book import BookIngestionOrchestrator
    if hasattr(BookIngestionOrchestrator, "_store_in_qdrant"):
        # Check for storage monitoring in the method
        import inspect
        source = inspect.getsource(BookIngestionOrchestrator._store_in_qdrant)
        if "WARNING" in source or "free-tier" in source.lower():
            compliance["free_tier"] = True

    # 4. OpenAI Agents SDK - Check agent structure
    from app.agents import retriever, selected_text
    if hasattr(retriever, "retriever_agent") and hasattr(selected_text, "selected_text_agent"):
        compliance["openai_agents_sdk"] = True

    # 5. Type Safety - Check mypy config
    import pyproject.toml
    with open("pyproject.toml") as f:
        content = f.read()
        if 'strict = true' in content:
            compliance["type_safety"] = True

    return compliance


if __name__ == "__main__":
    result = verify_compliance()
    print("Constitution Compliance Check:")
    for principle, passed in result.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {principle.replace('_', ' ').title()}: {status}")

    all_passed = all(result.values())
    print(f"\nOverall: {'✅ PASS' if all_passed else '❌ FAIL'}")
