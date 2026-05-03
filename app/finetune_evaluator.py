from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def query_finetune(question: str) -> dict:
    prompt = f"""You are an Australian healthcare assistant with deep knowledge 
of Medicare and PBS systems.
Answer the following question accurately and concisely.

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {
        "text": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens
    }