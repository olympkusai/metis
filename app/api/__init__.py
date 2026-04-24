from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router

def register_routes(app: FastAPI):
    app.include_router(chat_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
