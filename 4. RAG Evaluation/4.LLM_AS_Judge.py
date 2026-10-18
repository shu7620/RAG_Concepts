from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class CustomMetricJudge(BaseModel):
    tone_score: int = Field(description="Professionalism rating from 1 to 5")
    contains_disclaimer: bool = Field(description="True if mandatory financial disclaimer is present")
    feedback: str = Field(description="Actionable notes if criteria failed")

def custom_rag_evaluator(question: str, answer: str) -> CustomMetricJudge:
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CustomMetricJudge)
    
    prompt = f"""Evaluate the following RAG response based on custom business rules:
1. Tone must be professional and empathetic.
2. Must contain a disclaimer if providing financial advice or numbers.

Question: {question}
Answer: {answer}
"""
    return judge_llm.invoke(prompt)

# Test Custom Judge
eval_result = custom_rag_evaluator(
    question="What is the projected return on Project Alpha?",
    answer="Project Alpha projects a 15% ROI next quarter. Note: This is an internal estimate and not financial advice."
)

print("\nTone Score:", eval_result.tone_score)
print("Contains Disclaimer:", eval_result.contains_disclaimer)
print("Feedback:", eval_result.feedback)