from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

docs = [
    Document(page_content="Q3 Revenue grew by 15%", metadata={"department": "finance", "year": 2024}),
    Document(page_content="Q4 Revenue projected at 20%", metadata={"department": "finance", "year": 2025}),
    Document(page_content="New hiring policy updated", metadata={"department": "hr", "year": 2025}),
]

embeddings=OpenAIEmbeddings(model="text-embeddings-3-small")
vectorstore=Chroma.from_documents(docs,embeddings)

# Apply metadata filtering directly in similarity search using logical operators
filtered_results=vectorstore.similarity_search(
    query="What is the revenue growth?",
    k=2,
    filter={
        "$and":[
            {"department":{"$eq":"finance"}},
            {"year":{"$eq":2024}}
        ]
    }
)

for doc in filtered_results:
    print(f"Result: {doc.page_content} | Metadata: {doc.metadata}")