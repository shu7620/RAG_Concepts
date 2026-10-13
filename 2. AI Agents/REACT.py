import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.prebuilt import create_react_agent
from dotenv import load_dotenv

# 1. Define tools

@tool
def get_balance(user_id: str)->str:
    """Fetches the current account balance for a given user_id."""
    # Mock database call
    db = {"user_101": "$1,250.00", "user_102": "$42.50"}
    return db.get(user_id,"User ID not found.")

@tool
def calculate_tax(amount: float,tax_rate:float=0.18)->str:
    """Calculates tax on a specific dollar amount given a tax rate."""
    tax=amount*tax_rate
    return f"Calculated tax: ${tax:.2f} at a rate of {tax_rate * 100}%"

tools=[get_balance,calculate_tax]

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

# 2. Create REACT agent
agent=create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful banking assistant. Always use tools to verify balances and tax calculations."
)

# 3. Run the Agent
# Pass input as a message list in state
query = "How much tax would user_101 owe on their total balance at a 20% tax rate?"


print(f"--- Query ---\n{query}\n")
response=agent.invoke({"messages":[("user",query)]})


# Inspect the intermediate thought process and final response
print("--- Agent Loop History ---")
for message in response["messages"]:
    role=message.type.upper()
    content=message.content
    
    # Check if the LLM called a tool
    if hasattr(message,"tool_calls") and message.tool_calls:
        print(f"\n[{role} -> THOUGHT & ACTION]")
        for call in message.tool_calls:
            print(f"  Tool: {call['name']}({call['args']})")
    elif role=="TOOL":
        print(f"\n[{role} -> OBSERVATION]")
        print(f"  Output: {content}")
    else:
        print(f"\n[{role}]")
        print(f"  {content}")