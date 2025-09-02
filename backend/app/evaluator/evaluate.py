from app.evaluator.gemini import get_gemini_llm, get_gemini_embeddings
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision, answer_correctness, answer_relevancy
from dotenv import load_dotenv
import os
from datasets import Dataset

load_dotenv()

def evaluate_rag_with_gemini(rag_triad):
    gemini_llm = get_gemini_llm(
        model="gemini-2.0-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    gemini_embeddings = get_gemini_embeddings(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    results = evaluate(
        Dataset.from_dict(rag_triad),
        metrics=[faithfulness, context_recall, context_precision, answer_correctness, answer_relevancy],
        llm=gemini_llm,
        embeddings=gemini_embeddings
    )

    return results