# Evaluator- Optimizer Workflow
import os
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 1. Define State
class WorkflowState(TypedDict):
    topic: str
    draft: str
    feedback: str
    iteration: int
    is_approved: bool

# 2. Define Output Schema for Evaluation
class EvaluationResult(BaseModel):
    is_approved: bool = Field(description="True if quality standards are met, False otherwise")
    feedback: str = Field(description="Specific feedback or corrections needed if not approved")

# 3. Initialize Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
evaluator_llm = llm.with_structured_output(EvaluationResult)

# 4. Node Definitions
def generator_node(state: WorkflowState) -> dict:
    """Generates or revises the draft based on current state and feedback."""
    iteration = state.get("iteration", 0) + 1
    
    if state.get("feedback"):
        prompt = f"Topic: {state['topic']}\nPrevious Draft: {state['draft']}\nFeedback: {state['feedback']}\nRewrite the draft addressing all feedback."
    else:
        prompt = f"Write a concise, high-impact 2-sentence summary explaining: {state['topic']}"
        
    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": iteration}

def evaluator_node(state: WorkflowState) -> dict:
    """Evaluates the generated draft against quality standards."""
    prompt = f"""Evaluate this draft on the topic '{state['topic']}':
Draft: {state['draft']}

Criteria: Must be engaging, accurate, and under 40 words.
"""
    result: EvaluationResult = evaluator_llm.invoke(prompt)
    return {"is_approved": result.is_approved, "feedback": result.feedback}

# 5. Conditional Routing Logic
def route_approval(state: WorkflowState) -> Literal["generator", "__end__"]:
    """Routes execution based on evaluation and maximum iteration cap."""
    if state["is_approved"] or state["iteration"] >= 3:
        return END
    return "generator"

# 6. Build the LangGraph Workflow
builder=StateGraph(WorkflowState)

builder.add_node("generator",generator_node)
builder.add_node("evaluator",evaluator_node)

builder.add_edge(START,"generator")
builder.add_edge("generator","evaluator")

#Add conditional edges
builder.add_conditional_edges("evaluator",route_approval)

workflow=builder.compile()

# 7. Run Workflow
initial_state = {
    "topic": "Retrieval-Augmented Generation (RAG) in AI",
    "draft": "",
    "feedback": "",
    "iteration": 0,
    "is_approved": False
}

print("--- Running Agentic Workflow ---")
final_state = workflow.invoke(initial_state)

print(f"\nTotal Iterations: {final_state['iteration']}")
print(f"Approved: {final_state['is_approved']}")
print(f"\nFinal Draft:\n{final_state['draft']}")