import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Install HuggingFace Transformer
# pip install -U sentence-transformers

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_apple")

url = "https://www.apple.com/"

loader = WebBaseLoader(url)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)

if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store in {persistent_directory} ---")
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)
    
retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

query = "What new products are announced on Apple.com?"

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
