from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import qdrant_client
import openai
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
# Step 1: Connect to Qdrant
#client = QdrantClient(host="localhost", port=6333)  # Change host if using remote Qdrant




#openai.api_key = uncomment, give your OpenAI key here

#client = qdrant_client.QdrantClient(
#    url=qdrant_url,
#    port=6333,
#    api_key=qdrant_api_key,
#    timeout=30,
#)

embedding_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai.api_key )




# Step 2: Load Documents of Multiple Formats

documents = SimpleDirectoryReader(
        input_dir="d1/",  # Directory with mixed file formats
        recursive=True,  # Scan subfolders too
        encoding="ISO-8859-1"
    ).load_data()

# Step 3: Set Up Embeddings and Vector Store
embedding_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai.api_key)

index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)



# Step 5: Create Chat Engine
chat_engine = index.as_chat_engine(chat_mode="context", system_prompt="""
    You are a helpful chatbot that provides accurate and detailed answers. 
    You understand and respond based on the uploaded documents in various formats.
""")

# Step 6: Start the Chatbot
def chatbot():
    print("Chatbot started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        
        response = chat_engine.chat(user_input)
        print(f"Bot: {response}")

chatbot()



