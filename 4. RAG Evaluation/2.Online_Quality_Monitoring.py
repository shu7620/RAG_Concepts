# Instead of waiting for offline batch runs, production pipelines evaluate responses in real time before returning them to users. If the faithfulness score drops below a defined threshold, the pipeline intercepts the response to prevent returning potential hallucinations.

import os
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

# 1. Real-time Faithfulness Evaluator schema
class OnlineFaithfulnessScore(BaseModel):
    is_supported: bool = Field(description="True if every claim in the answer is backed by retrieved context")
    unsupported_claims: list[str] = Field(description="List of claims in answer not present in context", default_factory=list)
    confidence_score: float = Field(description="Score between 0.0 and 1.0 indicating degree of grounding")
    

def verify_grounding_online(context:str,answer:str)->OnlineFaithfulnessScore:
    evaluator=llm.with_structured_output(OnlineFaithfulnessScore)
    prompt = f"""Compare the generated answer against the retrieved context.
Determine if every statement in the answer is explicitly supported by the context.

Retrieved Context:
{context}

Generated Answer:
{answer}
"""
    return evaluator.invoke(prompt)

# 2. Simulated RAG Response Inspection
sample_context = "Project Alpha is led by Dr. Sarah Chen to develop autonomous underwater drones."
sample_answer_1 = "Dr. Sarah Chen leads Project Alpha for underwater drones."
sample_answer_2 = "Dr. Sarah Chen leads Project Alpha and the project has a $5 million budget." # Unbacked claim!

# test answer 1 (Grounded)
eval_1=verify_grounding_online(sample_context,sample_answer_1)
print("Answer 1 Grounded:", eval_1.is_supported, "| Confidence:", eval_1.confidence_score)

# Test Answer 2 (Contains Hallucination)
eval_2 = verify_grounding_online(sample_context, sample_answer_2)
print("Answer 2 Grounded:", eval_2.is_supported, "| Unsupported Claims:", eval_2.unsupported_claims)
