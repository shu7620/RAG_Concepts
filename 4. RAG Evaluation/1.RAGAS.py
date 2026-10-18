# Ragas evaluates RAG systems without needing human-annotated ground truth for every test query. It evaluates performance using four primary component metrics known as the RAG Triad
# Faithfulness(Is the LLM's answer derived only from the retrieved context?), 
# Answer Relevance(Does the generated answer directly address the user's question?), 
# Context Precision(Are relevant retrieved chunks ranked higher at the top of the vector search results?)
# and Context Recall(Did the retriever fetch all the necessary information needed to answer the question (compared against a reference ground truth)?).


import os 
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)
from dotenv import load_dotenv

load_dotenv()


# 1. Construct Evaluation Dataset
# Ragas requires a dataset containing: question, answer, contexts, and ground_truth
eval_data = {
    "question": [
        "Who leads Project Alpha and what is its goal?",
        "What is the battery life of the drone developed in Project Alpha?"
    ],
    "answer": [
        "Dr. Sarah Chen leads Project Alpha, which aims to develop autonomous underwater drones.",
        "The drone battery life is 24 hours of continuous underwater operation." # Hallucination (not in context)
    ],
    "contexts": [
        [
            "Project Alpha was launched in 2024 to develop autonomous underwater drones.",
            "The lead engineer on Project Alpha is Dr. Sarah Chen."
        ],
        [
            "Project Alpha was launched in 2024 to develop autonomous underwater drones.",
            "The lead engineer on Project Alpha is Dr. Sarah Chen."
        ]
    ],
    "ground_truth": [
        "Project Alpha is led by Dr. Sarah Chen to build autonomous underwater drones.",
        "Information regarding battery life is not specified in the documentation."
    ]
}

# Convert dictionary to Hugging Face Dataset format expected by Ragas
dataset=Dataset.from_dict(eval_data)

# 2. select metrics to evaluate
metrics=[
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
]

# 3. Run Evaluation
print("--- Running RAG Evaluation via Ragas ---")
results=evaluate(
    dataset=dataset,
    metrics=metrics
)

# 4. Inspect Evaluation Scores
df_results = results.to_pandas()
print("\n--- Summary Evaluation Table ---")
print(df_results[["question", "faithfulness", "answer_relevance", "context_precision", "context_recall"]])