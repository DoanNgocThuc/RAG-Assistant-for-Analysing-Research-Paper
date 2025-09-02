from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_gemini_llm(model="gemini-2.0-flash", api_key=None):
    """
    Trả về LLM tương thích với Ragas bằng LangChain wrapper.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0
    )
    return llm

def get_gemini_embeddings(api_key=None):
    """
    Trả về embeddings tương thích với Ragas bằng LangChain wrapper.
    """
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # model embedding của Gemini
        google_api_key=api_key
    )
    return gemini_embeddings

    
