import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

from langchain_core.document_loaders.base import Document

class UTF8TextLoader(TextLoader):
    def lazy_load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        yield Document(page_content=text, metadata={"source": self.file_path})
        


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "Chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent Directory does not exists. Initializing vector store...")
    
    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
        
    # loader = TextLoader(file_path)
    loader = UTF8TextLoader(file_path)

    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")
    
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = AzureOpenAIEmbeddings(
        model="Your AzureAI model name",
        azure_endpoint="Your AzureAI endpoint",
        api_key="Your AzureAI key",
        openai_api_version="2023-05-15"
    )
    print("\n--- Finished creating embeddings ---")
    
    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

