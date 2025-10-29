import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_service import rag_service

FRONTEND_URL = "http://localhost:5173"
DEFAULT_PORT = 8000

app = FastAPI(title="RAG PDF Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    confidence_score: int
    sources: list[dict]  # List of source dicts with page, section, preview

class ApiKeyRequest(BaseModel):
    api_key: str

stored_api_key = None

@app.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    global stored_api_key
    
    # Basic format validation - OpenAI keys start with sk-
    if not request.api_key or not request.api_key.startswith('sk-'):
        raise HTTPException(status_code=400, detail="Invalid API key format")
    
    # Validate by attempting to list models
    try:
        from openai import OpenAI
        test_client = OpenAI(api_key=request.api_key)
        test_client.models.list()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid API key: {str(e)}")
    
    stored_api_key = request.api_key
    rag_service.update_api_key(request.api_key)
    
    return {"message": "API key set successfully"}

@app.get("/health")
async def health_check():
    has_docs = rag_service.has_documents()
    return {"status": "healthy", "has_documents": has_docs}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if not stored_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not set. Please set API key first.")

    tmp_file_path = None
    try:
        # Ensure API key is set in the service
        if not rag_service.openai_client:
            raise HTTPException(status_code=400, detail="OpenAI API key not set. Please set API key first.")
        # Write uploaded file to temp location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        documents, extracted_text = rag_service.process_pdf(tmp_file_path)
        rag_service.add_documents(documents)
        
        result = {
            "message": f"Successfully processed {file.filename}",
            "chunks": len(documents),
            "filename": file.filename,
            "extracted_text": extracted_text
        }
        
        return result
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except RuntimeError as e:
        if "API key not set" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Cleanup temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass  # Ignore cleanup errors

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not rag_service.has_documents():
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    try:
        answer, confidence_score, sources = rag_service.query(request.question)
        return QueryResponse(
            answer=answer,
            confidence_score=confidence_score,
            sources=sources
        )
    except RuntimeError as e:
        if "API key not set" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/configuration")
async def get_configuration():
    try:
        config = rag_service.get_configuration()
        explanations = rag_service.get_parameter_explanations()
        return {
            "configuration": config,
            "explanations": explanations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@app.post("/configuration")
async def update_configuration(config_data: dict):
    try:
        updated_config = rag_service.update_configuration(config_data)
        explanations = rag_service.get_parameter_explanations()
        return {
            "message": "Configuration updated successfully",
            "configuration": updated_config,
            "explanations": explanations
        }
    except RuntimeError as e:
        if "API key not set" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)