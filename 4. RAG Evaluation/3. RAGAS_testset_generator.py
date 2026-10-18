# Creating manual (question, context, ground_truth) evaluation datasets is time-consuming. You can use Ragas to generate synthetic test datasets directly from raw document chunks automatically using multi-hop reasoning and query transformations.

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple,reasoning,multi_context
from dotenv import load_dotenv
load_dotenv()


# 1. Load Documents
docs = [
    "Project Alpha is led by Dr. Sarah Chen and focuses on autonomous underwater drones.",
    "Project Beta is directed by Mark Miller and develops high-altitude solar communication balloons.",
    "Project Alpha received the 2024 Innovation Award, whereas Project Beta started field trials in 2025."
]


# 2. Setup Generator Models
generator_llm=ChatOpenAI(model="gpt-4o-mini")
critic_llm=ChatOpenAI(model="gpt-4o-mini")
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Initialize Ragas Testset Generator
generator=TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings
)

# 4. Generate Synthetic Test Cases with Evolutionary Distributions
# (Simple questions, multi-context reasoning questions, and conditional queries)

distributions={
    simple: 0.5,
    reasoning:0.25,
    multi_context:0.25
}

testset=generator.generate_with_langchain_docs(
    documents=[TextLoader.from_documents(docs)],
    test_size=3,
    distributions=distributions
)

df=testset.to_pandas()
print("--- Synthetically Generated Test Dataset ---")
print(df[["question", "ground_truth"]])