# Supervisor Pattern in Langgraph
import os
from typing import Literal,TypedDict,Annotated
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# Shared state
class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    next_step: str

# Router Schema
class RouterResponse(BaseModel):
    next_agent: Literal["researcher","coder","FINISH"]
    
# Nodes
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_node(state: AgentState) -> dict:
    prompt = """You are a supervisor managing two agents:
1. 'researcher': Researches facts and concepts.
2. 'coder': Writes clean Python code.

Based on context, pick the next worker or choose 'FINISH' if complete."""
    messages=[("system",prompt)]+state["messages"]
    router=llm.with_structured_output(RouterResponse)
    result=router.invoke(messages)
    return {"next_step":result.next_agent}


def researcher_node(state: AgentState) -> dict:
    # Simulated search execution
    content = "Research Result: LangGraph uses directional state graphs to manage multi-agent loops."
    return {"messages": [HumanMessage(content=content, name="researcher")]}

def coder_node(state: AgentState) -> dict:
    # Simulated coding execution
    content = "Code Result: `builder = StateGraph(State)` initializes a graph structure."
    return {"messages": [HumanMessage(content=content, name="coder")]}


# Conditional Routing
def route_next(state:AgentState)->str:
    if state["next_step"]=="FINISH":
        return END
    return state["next_step"]

# Graph
builder=StateGraph(AgentState)
builder.add_node("supervisor",supervisor_node)
builder.add_node("researcher",researcher_node)
builder.add_node("coder",coder_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_next)
builder.add_edge("researcher", "supervisor")  # Workers report back to supervisor
builder.add_edge("coder", "supervisor")

graph=builder.compile()

# Invoke Workflow
res = graph.invoke({"messages": [("user", "Explain LangGraph state and show python code to initialize it.")]})
for msg in res["messages"]:
    print(f"[{msg.type.upper()} - {getattr(msg, 'name', 'User')}]: {msg.content}")