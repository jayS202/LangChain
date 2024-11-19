import os

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = AzureOpenAIEmbeddings(
    model="Your AzureAI model name",
    azure_endpoint="Your AzureAI endpoint",
    api_key="Your AzureAI key",
    openai_api_version="2023-05-15"
)


db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "How did Juliet die?"

retreiver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.1},
)

relevent_docs = retreiver.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevent_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")