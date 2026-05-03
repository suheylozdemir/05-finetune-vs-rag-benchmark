import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def measure_latency(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    latency = time.time() - start
    return result, latency

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    pricing = {
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
     }
    if model not in pricing:
        return 0.0
    cost = (prompt_tokens / 1000 * pricing[model]["prompt"]) + \
           (completion_tokens / 1000 * pricing[model]["completion"])
    return round(cost, 6)

def evaluate_accuracy(question: str, predicted_answer: str, ground_truth: str) -> float:
    prompt = f"""You are an objective evaluator. Score the predicted answer against the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted_answer}

Score from 0.0 to 1.0 where:
1.0 = completely correct
0.5 = partially correct
0.0 = completely wrong

Respond with ONLY a number between 0.0 and 1.0."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

def run_benchmark(questions: list, answer_func, model: str) -> list:
    results = []
    
    for q in questions:
        answer, latency = measure_latency(answer_func, q["question"])
        
        accuracy = evaluate_accuracy(
            q["question"],
            answer["text"],
            q["ground_truth"]
        )
        
        cost = calculate_cost(
            answer.get("prompt_tokens", 0),
            answer.get("completion_tokens", 0),
            model
        )
        
        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "predicted": answer["text"],
            "ground_truth": q["ground_truth"],
            "accuracy": accuracy,
            "latency": round(latency, 3),
            "cost": cost
        })
    
    return results