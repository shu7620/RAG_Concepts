import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# 1. create sample docs
with open("sample_doc.txt","w") as f:
    f.write(
        "Project Alpha was launched in 2024 to develop autonomous underwater drones. "
        "The lead engineer on Project Alpha is Dr. Sarah Chen. "
        "Project Beta focuses on high-altitude solar communication balloons led by Mark Miller."
    )
    


# 2. Load documents
loader=TextLoader("sample_doc.txt")
docs=loader.load()

# 3. Chunking
text_splitter=RecursiveCharacterTextSplitter(separators=['.'],chunk_size=50,chunk_overlap=5)
chunks=text_splitter.split_documents(docs)


# 4. Embedding
embeddings=OpenAIEmbeddings(model="text-embeddings-3-small")

# 5.Vector store
vectorstore=Chroma.from_documents(documents=chunks,embedding=embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k":2})

# 6. RAG chain(LCEL)
prompt_template="""Answer the following question based on the following context {context}
 Question: {question}
 Answer: ""
"""

prompt=ChatPromptTemplate.from_template(prompt_template)
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context":retriever | format_docs,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



# 7. Test the pipeline
question = "Who leads Project Alpha and what is its goal?"
response=rag_chain.invoke(question)

print(f"--- Question ---\n{question}\n")
print(f"--- Response ---\n{response}")