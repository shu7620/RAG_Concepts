import os
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()


# 1. Define the Agent State
class AgentState(TypedDict):
    # 'add_messages' ensures new messages are appended rather than overwriting existing history
    messages: Annotated[List[BaseMessage], add_messages]
    

# 2. Define the LLM Node
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

def chatbot_node(state:AgentState):
    response=llm.invoke(state["messages"])
    return {"messages":[response]}

# 3. Build Graph with checkpointer
builder=StateGraph(AgentState)
builder.add_node("chatbot",chatbot_node)
builder.add_edge(START,"chatbot")
builder.add_edge("chatbot",END)

# InMemorySaver acts as the agent's short-term memory store
memory=InMemorySaver()
graph=builder.compile(checkpointer=memory)


# 4. Run Multi-turn Conversation using thread_id
config={"configurable":{"thread_id":"session_101"}}


# Turn 1
input_1 = {"messages": [("user", "Hi! My name is Shubham.")]}
out_1=graph.invoke(input_1,config)
print("Agent:", out_1["messages"][-1].content)

# Turn 2 (Agent remembers the previous turn via thread_id)
input_2 = {"messages": [("user", "What is my name?")]}
out_2 = graph.invoke(input_2, config)
print("Agent:", out_2["messages"][-1].content)