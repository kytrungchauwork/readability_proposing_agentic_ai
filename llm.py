from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        openai_api_key="API-KEY",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.2
    )