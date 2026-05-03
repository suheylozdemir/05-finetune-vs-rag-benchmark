import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app.rag_pipeline import index_documents, query_rag
from app.finetune_evaluator import query_finetune
from app.metrics import run_benchmark

def load_data():
    with open("data/healthcare_qa.json", "r") as f:
        data = json.load(f)
    return data["documents"], data["questions"]

def save_results(rag_results: list, finetune_results: list):
    os.makedirs("results", exist_ok=True)

    with open("results/benchmark_results.json", "w") as f:
        json.dump({
            "rag": rag_results,
            "finetune": finetune_results
        }, f, indent=2)

def generate_report(rag_results: list, finetune_results: list):
    rag_df = pd.DataFrame(rag_results)
    ft_df = pd.DataFrame(finetune_results)

    summary = {
        "RAG": {
            "avg_accuracy": round(rag_df["accuracy"].mean(), 3),
            "avg_latency": round(rag_df["latency"].mean(), 3),
            "total_cost": round(rag_df["cost"].sum(), 6)
        },
        "Fine-tune": {
            "avg_accuracy": round(ft_df["accuracy"].mean(), 3),
            "avg_latency": round(ft_df["latency"].mean(), 3),
            "total_cost": round(ft_df["cost"].sum(), 6)
        }
    }

    print("\n===== BENCHMARK RESULTS =====")
    for system, metrics in summary.items():
        print(f"\n{system}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fine-tune vs RAG Benchmark — Australian Healthcare", fontsize=14)

    metrics = ["accuracy", "latency", "cost"]
    titles = ["Accuracy (higher is better)", "Latency in seconds (lower is better)", "Cost in USD (lower is better)"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        data = {
            "System": ["RAG", "Fine-tune"],
            metric: [
                summary["RAG"][f"avg_{metric}"] if metric != "cost" else summary["RAG"]["total_cost"],
                summary["Fine-tune"][f"avg_{metric}"] if metric != "cost" else summary["Fine-tune"]["total_cost"]
            ]
        }
        df = pd.DataFrame(data)
        sns.barplot(data=df, x="System", y=metric, hue="System", ax=axes[i], palette="Blues_d", legend=False)
        axes[i].set_title(title)
        axes[i].set_xlabel("")

    plt.tight_layout()
    plt.savefig("results/benchmark_chart.png", dpi=150)
    print("\nChart saved to results/benchmark_chart.png")

    return summary

def main():
    print("Loading data...")
    documents, questions = load_data()

    print("Indexing documents into Pinecone...")
    index_documents(documents)

    print("Running RAG benchmark...")
    rag_results = run_benchmark(questions, query_rag, "gpt-4.1-mini")

    print("Running Fine-tune benchmark...")
    finetune_results = run_benchmark(questions, query_finetune, "gpt-4.1-mini")

    save_results(rag_results, finetune_results)
    generate_report(rag_results, finetune_results)

if __name__ == "__main__":
    main()