import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Pydantic Schemas for Structured Plan Outputs
class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to achieve the goal in order")

class ReplanningOutput(BaseModel):
    is_complete: bool = Field(description="True if original goal is fully achieved")
    response: str = Field(description="Final response to give to user if complete", default="")
    remaining_steps: List[str] = Field(description="Updated remaining steps if incomplete", default_factory=list)
    
    
# 2. Plan-and-Execute Workflow Implementation
def run_plan_and_execute(goal:str):
    # Step A: Generate Initial Plan
    planner=llm.with_structured_output(Plan)
    initial_plan: Plan=planner.invoke(f"Create a step-by-step plan to answer: {goal}")
    
    steps=initial_plan.steps
    completed_steps=[]
    print(f"📋 Initial Plan Generated ({len(steps)} steps):", steps)
    
    iteration=0
    while steps and iteration<5:
        iteration+=1
        current_step=steps[0]
        print(f"\n⚡ Executing Step {iteration}: '{current_step}'")
        
        # Step B: Executor executes the current step
        execution_prompt = f"Goal: {goal}\nCompleted Steps: {completed_steps}\nCurrent Step to execute: {current_step}\nProvide result for this step:"
        step_result=llm.invoke(execution_prompt).content
        print(f"  Result: {step_result[:100]}...")
        
        completed_steps.append(f"Step: {current_step} -> Output: {step_result}")
        
        
        # Step C: Re-planner inspects progress and updates remaining plan
        replanner = llm.with_structured_output(ReplanningOutput)
        replan_prompt = f"""Original Goal: {goal}
Executed Steps Progress: {completed_steps}
Remaining Steps left in queue: {steps[1:]}

Assess if the goal is completely satisfied or update the remaining steps required."""
        
        replan:ReplanningOutput=replanner.invoke(replan_prompt)
        
        if replan.is_complete:
            print("\n🎯 Goal Satisfied!")
            return replan.response
        else:
            steps=replan.remaining_steps
            print(f"  Plan Updated. Remaining steps: {steps}")
            

  # Test
run_plan_and_execute("Compare the main differences between Python lists and tuples and give a code snippet for each.")      