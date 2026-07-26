from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_core import ask_spezi

app = FastAPI(
    title="Spezi AI Backend API",
    description="REST API for the Spezi German Language Tutor Agent",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    response: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Spezi API"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        reply = ask_spezi(user_id=request.user_id, message=request.message)
        return ChatResponse(user_id=request.user_id, response=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
