from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Vector store and Record Manager (using SQLite to track document state)

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore=Chroma(collection_name="managed_docs",embedding_function=embeddings)

record_manager=SQLRecordManager(
    namespace="chroma/managed_docs",
    db_url="sqlite:///record_manager.sql"
)

# Initialize the record manager database
record_manager.create_schema()

# 2. Initial Document Batch
doc1 = Document(page_content="Original content for document 1", metadata={"source": "doc1.txt"})
doc2 = Document(page_content="Original content for document 2", metadata={"source": "doc2.txt"})

# Index initial documents
indexing_result=index(
    [doc1,doc2],
    record_manager,
    vectorstore,
    cleanup="incremental", # Automatically cleans old versions of modified docs
    source_id_key="source"
)
print("Initial Indexing Result:", indexing_result)

# 3. Update doc1 and keep doc2 unchanged
doc1_updated = Document(page_content="UPDATED content for document 1", metadata={"source": "doc1.txt"})

# Re-run indexer
indexing_result_2=index(
    [doc1_updated,doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source"
)

print("Updated Indexing Result:", indexing_result_2)
# Output will show 1 updated, 1 skipped (doc2 wasn't re-embedded!)