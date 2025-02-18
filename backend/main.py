from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import numpy as np
import faiss
import pymupdf 
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-system-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text.strip()

def chunk_text(text, max_chars=8000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def get_embedding(text):
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        try:
            response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
            embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error generating embedding: {e}")

    if embeddings:
        return np.mean(np.array(embeddings), axis=0) 
    else:
        return np.zeros(1536) 


PDF_DIR = "../pdf-files"  

pdf_reports = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        content = extract_text_from_pdf(pdf_path)
        
        # Truncate text if too long (for safety)
        if len(content) > 8000:
            print(f"Warning: {filename} text too long ({len(content)} characters), truncating...")
        
        pdf_reports.append({
            "id": len(pdf_reports) + 1, 
            "title": filename, 
            "content": content[:8000]  
        })

# Create FAISS index for semantic search
dimension = 1536  # OpenAI embedding dimension
index = faiss.IndexFlatL2(dimension)

if pdf_reports:
    report_embeddings = [get_embedding(report["content"]) for report in pdf_reports]
    report_embeddings = np.array(report_embeddings, dtype=np.float32)  # Ensure correct shape
    index.add(report_embeddings)

class QueryRequest(BaseModel):
    query: str

@app.get("/reports")
def get_reports():
    return pdf_reports

@app.post("/query")
async def process_query(request: QueryRequest):
    query = request.query

    
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, k=2)  # Retrieve top 2 reports
    relevant_info = [pdf_reports[idx]["content"] for idx in indices[0]]

   
    if not relevant_info:
        relevant_info = ["No relevant information found in the reports."]

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
        "sources": [pdf_reports[idx] for idx in indices[0]],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
