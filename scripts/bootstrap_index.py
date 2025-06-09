from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
import os

try:
    # Set the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")  # Aligned with main script
    print("Embedding model initialized: all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to initialize HuggingFaceEmbedding: {str(e)}")
    raise

try:
    # Load documents from data/docs
    documents = SimpleDirectoryReader("data/docs").load_data()
    if not documents:
        print("No documents found in data/docs")
        raise ValueError("No documents to index")
    print(f"Loaded {len(documents)} documents from data/docs")
except Exception as e:
    print(f"Failed to load documents: {str(e)}")
    raise

try:
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    print("Index created successfully")
except Exception as e:
    print(f"Failed to create index: {str(e)}")
    raise

try:
    # Persist index to data/llama_index
    persist_path = os.path.join(os.path.dirname(__file__), "..", "data", "llama_index")
    os.makedirs(persist_path, exist_ok=True)
    index.storage_context.persist(persist_path)
    print(f"Index persisted to {persist_path}")
except Exception as e:
    print(f"Failed to persist index: {str(e)}")
    raise
