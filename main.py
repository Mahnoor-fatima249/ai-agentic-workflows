import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_application_env

# Environment variables load karein (API Keys ke liye)
load_application_env()

app = FastAPI(
    title="High-Performance AI Backend",
    description="Production-ready FastAPI backend integrating Gemini API for Agentic Workflows.",
    version="1.0.0"
)

# Gemini Client Initialize karein (Latest Google GenAI SDK)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in environment variables.")
    return genai.Client(api_key=api_key)

# Input Validation ke liye Pydantic Model
class ChatRequest(BaseModel):
    user_message: str = Field(..., example="Explain the core concept of Multi-Agent Workflows.")
    system_instruction: str = Field(
        default="You are an expert AI & Backend Architect. Give concise, highly technical responses.",
        example="You are a helpful assistant."
    )

# AI Chat Generation Endpoint
@app.post("/api/v1/chat", tags=["AI Generation"])
async def generate_ai_response(request: ChatRequest, client: genai.Client = Depends(get_gemini_client)):
    try:
        # Gemini 2.5 Flash use kar rahe hain high-speed backend execution ke liye
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.user_message,
            config=types.GenerateContentConfig(
                system_instruction=request.system_instruction,
                temperature=0.3, # Low temperature for factual and structured backend logic
            )
        )
        
        return {
            "status": "success",
            "model_used": "gemini-2.5-flash",
            "response": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

# Health Check Endpoint (For Cloud Deployment validation)
@app.get("/health", tags=["System Health"])
async def health_check():
    return {"status": "healthy", "service": "AI & Agentic Backend Layer"}
