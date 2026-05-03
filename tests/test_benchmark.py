import pytest
from unittest.mock import patch, MagicMock
from app.metrics import calculate_cost, run_benchmark

def test_calculate_cost_gpt4():
    cost = calculate_cost(1000, 500, "gpt-4.1-mini")
    assert cost > 0

def test_calculate_cost_unknown_model():
    cost = calculate_cost(1000, 500, "unknown-model")
    assert cost == 0.0

def test_calculate_cost_returns_float():
    cost = calculate_cost(500, 200, "gpt-4.1-mini")
    assert isinstance(cost, float)

def test_run_benchmark_result_structure():
    mock_answer_func = MagicMock(return_value={
        "text": "The Medicare Levy is 2% of taxable income.",
        "prompt_tokens": 100,
        "completion_tokens": 50
    })

    questions = [
        {
            "id": "q_001",
            "question": "What is the Medicare Levy rate?",
            "ground_truth": "The Medicare Levy is 2% of taxable income."
        }
    ]

    with patch("app.metrics.evaluate_accuracy", return_value=0.9):
        results = run_benchmark(questions, mock_answer_func, "gpt-4.1-mini")

    assert len(results) == 1
    assert "accuracy" in results[0]
    assert "latency" in results[0]
    assert "cost" in results[0]
    assert "question_id" in results[0]

def test_run_benchmark_accuracy_range():
    mock_answer_func = MagicMock(return_value={
        "text": "Some answer.",
        "prompt_tokens": 100,
        "completion_tokens": 50
    })

    questions = [
        {
            "id": "q_001",
            "question": "What is bulk billing?",
            "ground_truth": "Bulk billing is when a doctor accepts Medicare benefit as full payment."
        }
    ]

    with patch("app.metrics.evaluate_accuracy", return_value=0.7):
        results = run_benchmark(questions, mock_answer_func, "gpt-4.1-mini")

    assert 0.0 <= results[0]["accuracy"] <= 1.0

def test_run_benchmark_latency_positive():
    mock_answer_func = MagicMock(return_value={
        "text": "Some answer.",
        "prompt_tokens": 100,
        "completion_tokens": 50
    })

    questions = [
        {
            "id": "q_001",
            "question": "What is PBS?",
            "ground_truth": "PBS is the Pharmaceutical Benefits Scheme."
        }
    ]

    with patch("app.metrics.evaluate_accuracy", return_value=0.8):
        results = run_benchmark(questions, mock_answer_func, "gpt-4.1-mini")

    assert results[0]["latency"] >= 0