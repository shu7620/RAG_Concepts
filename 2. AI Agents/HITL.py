import uuid
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver

class WorkflowState(TypedDict):
    amount: float
    recipient: str
    status: str
    
# Node 1: Initiate Transfer
def initiate_transfer(state: WorkflowState):
    print(f"Initiating transfer of ${state['amount']} to {state['recipient']}...")
    return {}


# Node 2: Pause for Human Approval
def human_approval_node(state: WorkflowState):
    # interrupt() halts graph execution and yields control back to caller
    approval=interrupt({
        "question": f"Approve payment of ${state['amount']} to {state['recipient']}?",
        "details":state
    })
    
    if approval.get("approved") is True:
        return {"status": "TRANSFER_COMPLETED"}
    else:
        return {"status":"TRANSFER_REJECTED"}
    
# construct graph
builder=StateGraph(WorkflowState)
builder.add_node("initiate",initiate_transfer)
builder.add_node("approval",human_approval_node)

builder.add_edge(START,"initiate")
builder.add_edge("initiate","approval")
builder.add_edge("approval",END)

memory=InMemorySaver()
graph=builder.compile(checkpointer=memory)

# Run execution up to the interrupt point
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
initial_payload = {"amount": 5000.00, "recipient": "Acme Corp", "status": "PENDING"}

step_1=graph.invoke(initial_payload,config)
print("\n[PAUSED] Interrupt trigger received:", step_1.get("__interrupt__"))

# Simulate Human Reviewer approving the action via Command(resume=...)
print("\n[HUMAN ACTION] Approving payment...")
final_step=graph.invoke(Command(resume={"approved":True}),config)
print("Final Workflow Status:", final_step["status"])