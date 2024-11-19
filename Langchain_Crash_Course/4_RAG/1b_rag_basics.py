import os

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
Persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = AzureOpenAIEmbeddings(
    model="Your AzureAI model name",
    azure_endpoint="Your AzureAI endpoint",
    api_key="Your AzureAI key",
    openai_api_version="2023-05-15"
)

query = "Who is Odysseus' wife?"

db = Chroma(persist_directory=Persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.3},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")