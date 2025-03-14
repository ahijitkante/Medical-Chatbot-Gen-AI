from src.helper import load_pdf_file, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Retrieve API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key is missing! Set it in the .env file.")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key is missing! Set it in the .env file.")

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    google_api_key=GEMINI_API_KEY
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Create Pinecone index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Pinecone index '{index_name}' created successfully!")
else:
    print(f"Pinecone index '{index_name}' already exists.")

# Load PDF data and split into chunks
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)

# Batch process embeddings to avoid hitting quota limits
batch_size = 5  # Adjust based on API limits
total_chunks = len(text_chunks)

print(f"Processing {total_chunks} chunks in batches of {batch_size}...")

batched_embeddings = []
for i in range(0, total_chunks, batch_size):
    batch = text_chunks[i : i + batch_size]
    
    try:
        batch_embeddings = embeddings.embed_documents([chunk.page_content for chunk in batch])
        batched_embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(total_chunks//batch_size) + 1}")

        time.sleep(2)  # Wait to avoid rate limits

    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        time.sleep(10)  # Wait before retrying

# Store embeddings in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Embeddings successfully stored in Pinecone!")
