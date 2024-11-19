import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

# Install HuggingFace Transformer
# pip install -U sentence-transformers

load_dotenv()

current_dir = os.path.dirname(__file__)
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "How did Juliet die?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3, "score_threshold": 0.2},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}")
    
model = AzureChatOpenAI(
    model="Your model name",
    azure_endpoint="Your AzureAI model endpoint",
    api_key="Your AzureAI model Key",
    api_version="Your Azure API version",
    model_version="Your AzureAI model version"
)

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content=combined_input)
]
    
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)