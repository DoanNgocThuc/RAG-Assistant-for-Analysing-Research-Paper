# app/rag/prompts.py
from app.rag.generation import generate_with_ollama

def _build_system_prompt(mode: str):
    if mode.lower() == "novice":
        return "You are an assistant that explains scientific papers to a novice..."
    elif mode.lower() == "reviewer":
        return "You are an assistant that helps peer reviewers..."
    elif mode.lower() == "normal":
        return "Keep it simple and avoid unnecessary jargon."
    else:
        return "You are an assistant that explains papers to an informed researcher."

def explain_context(question: str, snippet: str):
    system_prompt = "You are an assistant that explains context..."
    user_prompt = f"Question: {question}\n\nSnippet: {snippet}\n\nExplain relevance."
    try:
        return generate_with_ollama(system_prompt, user_prompt, max_tokens=500)
    except Exception as e:
        return f"Failed to generate explanation: {e}"
