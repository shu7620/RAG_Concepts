from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# 1. Sample corpus with technical terms and codes
documents = [
    Document(page_content="Error code ERR-404 occurs when the server cannot find the requested resource."),
    Document(page_content="HTTP 404 Not Found is a standard response code in computer network communications."),
    Document(page_content="System crashes during heavy traffic spikes due to memory leakage in module B.")
]

# 2. Sparse Retriever (BM25 for exact keyword matching)
bm25_retriever=BM25Retriever.from_documents(documents)
bm25_retriever.k=2


# 3. Dense Retriever (Vector store for semantic search)
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore=Chroma.from_documents(documents,embeddings)
vector_retriever=vectorstore.as_retriever(search_kwargs={"k":2})


# 4. Ensemble Retriever (Hybrid Search combining BM25 + Vector Search)
hybrid_retriever=EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights=[0.5,0.5] # equal weight to keywords and semantic meaning
)

# Test query with exact error code
results = hybrid_retriever.invoke("What does ERR-404 mean?")
for doc in results:
    print(f"- {doc.page_content}")