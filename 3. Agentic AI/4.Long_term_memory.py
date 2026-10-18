import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

# 1. Setup Persistent Vector Memory Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
memory_store=Chroma(collection_name="agent_user_memory",embedding_function=embeddings)

# 2. Function to Save a Memory
def save_memory(fact:str,user_id:str):
    doc=Document(page_content=fact,metadata={"user_id":user_id})
    memory_store.add_documents([doc])
    print(f"🧠 Memory Stored: '{fact}' for {user_id}")
    
# 3. Function to Retrieve Memories
def recall_memories(query:str,user_id:str,k:int=2)->str:
    results=memory_store.similarity_search(
        query,k=k,filter={"user_id":{"$eq":user_id}}
    )
    memories=[doc.page_content for doc in results]
    return "\n".join(memories)

# Seed Memories
save_memory("User prefers answers in short bullet points", user_id="user_101")
save_memory("User works primarily with Python and FastAPI", user_id="user_101")

# 4. Agent Execution with Memory Injection
def memory_aware_agent(user_query:str,user_id:str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Retrieve relevant memories
    relevant_memories=recall_memories(user_query,user_id)
    
    system_prompt = f"""You are a helpful assistant.
User Profile / Memory History:
{relevant_memories}

Answer the user's question adhering strictly to their remembered preferences."""

    messages = [("system", system_prompt), ("user", user_query)]
    response = llm.invoke(messages)
    return response.content

# Test Memory Recall
print("\n--- Agent Response with Memory ---")
answer = memory_aware_agent("How should I build an API for user authentication?", user_id="user_101")
print(answer)