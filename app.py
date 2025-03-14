from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # ✅ Using GEMINI_API_KEY

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Set it in your .env file!")

# Initialize Gemini embeddings (Free model)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define Pinecone index name
index_name = "medicalbot"

# Connect to existing Pinecone index
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    raise RuntimeError(f"❌ Error connecting to Pinecone: {str(e)}")

# Initialize Free Gemini LLM (1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)

# Define chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create retrieval and response chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Route for home page
@app.route("/")
def index():
    return render_template('chat.html')

# Route for chatbot interaction
@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"error": "❌ Empty input received!"})

        print(f"📝 User Input: {msg}")
        response = rag_chain.invoke({"input": msg})

        answer = response.get("answer", "⚠️ No response generated.")
        print(f"🤖 Chatbot Response: {answer}")

        return jsonify({"response": answer})

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": "❌ Something went wrong. Please try again!"})

# Run Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
