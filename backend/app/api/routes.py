from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.llm import LLMEngine

router = APIRouter()
llm_engine = LLMEngine()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    use_speculative: Optional[bool] = True

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        response = await llm_engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            use_speculative=request.use_speculative
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available models and their configurations"""
    return {
        "models": [
            {
                "id": "gpt2",
                "name": "GPT-2",
                "description": "Small GPT-2 model for testing",
                "context_length": 1024
            },
            {
                "id": "gpt2-medium",
                "name": "GPT-2 Medium",
                "description": "Medium GPT-2 model for target model in speculative decoding",
                "context_length": 1024
            }
        ]
    } 