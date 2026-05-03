import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "healthcare-benchmark"

def initialize_pinecone():
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def index_documents(documents: list):
    index = initialize_pinecone()
    vectors = []
    for doc in documents:
        embedding = get_embedding(doc["content"])
        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                "title": doc["title"],
                "content": doc["content"]
            }
        })
    index.upsert(vectors=vectors)

def query_rag(question: str) -> dict:
    index = initialize_pinecone()
    question_embedding = get_embedding(question)
    
    results = index.query(
        vector=question_embedding,
        top_k=2,
        include_metadata=True
    )
    
    context = "\n".join([
        match["metadata"]["content"] 
        for match in results["matches"]
    ])
    
    prompt = f"""You are a helpful Australian healthcare assistant.
Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer concisely and accurately."""

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