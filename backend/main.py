from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import numpy as np
import faiss
import fitz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-system-1.onrender.com"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configurable PDF Directory
PDF_DIR = os.getenv("PDF_DIR", os.path.abspath(os.path.join(os.getcwd(), "..", "pdf-files")))

print("📂 Checking PDF files in:", PDF_DIR)
print("Found PDFs:", os.listdir(PDF_DIR))

# FAISS Index File (Now stored outside backend for persistence)
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "..", "faiss_index.bin")

# Define embedding dimensions (OpenAI "text-embedding-ada-002" = 1536)
DIMENSION = 1536

# ✅ Root Endpoint to Fix "404 Not Found"
@app.get("/")
def home():
    return {"message": "API is running! Go to /docs to test."}

# ✅ Ensure the PDF directory exists
if not os.path.exists(PDF_DIR):
    print(f"⚠️ Warning: PDF directory '{PDF_DIR}' not found. Creating it...")
    os.makedirs(PDF_DIR)

# ✅ Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text.strip()

# ✅ Chunk text for embedding processing
def chunk_text(text, max_chars=8000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# ✅ Generate OpenAI embeddings (Handles API failures)
def get_embedding(text):
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        try:
            response = openai.Embedding.create(  # ✅ Correct OpenAI method
                model="text-embedding-ada-002",
                input=[chunk]  # ✅ Must be a list
            )
            embeddings.append(response["data"][0]["embedding"])  # ✅ Correct response format

        except openai.OpenAIError as e:  # ✅ Correct error handling
            print(f"OpenAI API Error: {e}")

    return np.mean(np.array(embeddings), axis=0) if embeddings else np.zeros(DIMENSION)

# ✅ Load and process PDF reports
pdf_reports = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        content = extract_text_from_pdf(pdf_path)
        pdf_reports.append({
            "id": len(pdf_reports) + 1,
            "title": filename,
            "content": content[:8000]  # Truncate long texts
        })

if not pdf_reports:
    print("⚠️ Warning: No PDFs found in 'pdf-files' directory. Please upload PDFs!")

# ✅ Load or create FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)  # Load existing FAISS index
else:
    index = faiss.IndexFlatL2(DIMENSION)  # Create new FAISS index

if pdf_reports:
    report_embeddings = [get_embedding(report["content"]) for report in pdf_reports]
    report_embeddings = np.array(report_embeddings, dtype=np.float32)
    faiss.normalize_L2(report_embeddings)  # Normalize for better similarity search
    index.add(report_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)  # Save index to disk

# ✅ API Model
class QueryRequest(BaseModel):
    query: str

# ✅ Get available reports
@app.get("/reports")
def get_reports():
    if not pdf_reports:
        return {"message": "No reports found. Please upload PDFs to 'pdf-files'."}
    return pdf_reports

# ✅ Process Query (Search FAISS + Generate AI Response)
@app.post("/query")
async def process_query(request: QueryRequest):
    query = request.query

    if not pdf_reports:
        raise HTTPException(status_code=400, detail="No PDFs found. Upload reports first.")

    try:
        query_embedding = np.array([get_embedding(query)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)  # Optimize search
        distances, indices = index.search(query_embedding, k=2)  # Get top 2 results

        relevant_info = [pdf_reports[idx]["content"] for idx in indices[0] if idx < len(pdf_reports)]
        if not relevant_info:
            relevant_info = ["No relevant information found in the reports."]

        # Generate AI response
        prompt = f"Query: {query}\nRelevant Information: {', '.join(relevant_info[:500])}\nGenerate a concise and insightful response:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping analyze reports."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return {
            "response": response.choices[0].message["content"].strip(),
            "sources": [pdf_reports[idx] for idx in indices[0] if idx < len(pdf_reports)],
        }

    except openai.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {e}")

# ✅ Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
