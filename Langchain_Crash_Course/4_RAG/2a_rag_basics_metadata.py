import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.document_loaders.base import Document


class UTF8TextLoader(TextLoader):
    def lazy_load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        yield Document(page_content=text, metadata={"source": self.file_path})

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
        
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
    
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir,book_file)
        loader = UTF8TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
            
    textSplitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = textSplitter.split_documents(documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    
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


        