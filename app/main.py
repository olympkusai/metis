from fastapi import FastAPI
from app.api import register_routes

app = FastAPI(title="Crypto Agent")

register_routes(app)

@app.get("/")
async def root():
    return {"message": "Crypto Agent API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
