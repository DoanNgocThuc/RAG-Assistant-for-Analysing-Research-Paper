import json
from app.evaluator.evaluate import evaluate_rag_with_gemini
from app.rag.pipeline import process_question
from app.rag.generation import generate_with_ollama


def evaluate_RAG (question_groundtruth:list, pdf_path:str):
    print("Evaluating RAG...")
    answers = []
    questions = []
    reference_contexts_list = []
    contexts = []
    references = []
    for item in question_groundtruth:
        # Extract question
        question = item["question"]
        questions.append(question)

        # Extract references
        reference_contexts = item["groundtruth_chunks"]
        reference_contexts_list.append(reference_contexts)

        # Extract ground truth answer
        groundtruth_answer = item["groundtruth_answer"]
        references.append(groundtruth_answer)

        # Process the question -> answer, sources
        generated_answer,sources = process_question(question, mode="normal", pdf_path=pdf_path, k=3)

        # Append the generated answer to the answers list
        answers.append(generated_answer)

        # Append the sources to the contexts list
        snippets = []
        snippets = [item["snippet"] for item in sources]
        contexts.append(snippets)

        # Compare the generated answer with the ground truth
        # Return a score between 0 and 1

    evaluation_triad ={
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "reference": references,
        "reference_contexts": reference_contexts_list
    }
    score = evaluate_rag_with_gemini(rag_triad = evaluation_triad)
    return score


# def evaluate_answer_relevancy(evaluation_triad):
#     """
#     Đánh giá độ liên quan của answer với question cho 1 QA pair.
#     Args:
#         evaluation_triad: dict với keys "question", "contexts", "answer"
#     Returns:
#         relevance_score (float between 0 and 1)
#     """
#     print("Evaluating answer relevancy...")
#     question = evaluation_triad["question"][0]
#     answer = evaluation_triad["answer"][0]
#     contexts = evaluation_triad["contexts"][0]  # list các context

#     system_prompt = """You are an expert evaluator for question answering systems.
# Your task is to evaluate how relevant and responsive the answers are to their questions.
# Focus on whether the answer directly addresses what was asked.

# Score relevance using these exact criteria:
# 0 = Answer is completely off-topic or unrelated to the question
# 0.1 - 0.3 = Answer barely relates to the question's topic
# 0.4 - 0.6 = Answer partially addresses the question but misses key aspects
# 0.7 - 0.9 = Answer is mostly relevant and addresses the question well, but could be more focused
# 1 = Answer perfectly matches the question's requirements with excellent focus

# IMPORTANT: Return ONLY valid JSON with scores and explanations.
# Do not include any other text before or after the JSON."""

#     user_prompt = f"""
# Evaluate the relevance of this answer to its question:

# Question: {question}
# Answer: {answer}
# Contexts: {json.dumps(contexts, ensure_ascii=False, indent=2)}

# Analyze:
# 1. How directly the answer addresses the specific question asked
# 2. Whether the answer includes unnecessary or off-topic information
# 3. Whether the answer covers all aspects of the question
# 4. The focus and precision of the answer

# Return your evaluation in this exact JSON format:
# {{
#     "relevance_score": score_between_0_and_1,
#     "explanation": "Detailed explanation of the score",
#     "addressed_aspects": ["List aspects of the question that were addressed"],
#     "missing_aspects": ["List aspects of the question that were not addressed"],
#     "off_topic_content": ["List any irrelevant or unnecessary content"]
# }}
# """

#     result = generate_with_ollama(system_prompt, user_prompt)
#     result = result.strip()
#     if not result.startswith('{'):
#         start_idx = result.find('{')
#         if start_idx == -1:
#             raise ValueError("No JSON object found in response")
#         result = result[start_idx:]
#     if not result.endswith('}'):
#         end_idx = result.rfind('}')
#         if end_idx == -1:
#             raise ValueError("No closing brace found in response")
#         result = result[:end_idx + 1]

#     evaluation = json.loads(result)
#     return evaluation["relevance_score"]