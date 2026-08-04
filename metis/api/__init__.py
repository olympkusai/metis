from fastapi import FastAPI
from metis.api.chat import router as chat_router
from metis.api.conversations import router as conversations_router

def register_routes(app: FastAPI):
    app.include_router(chat_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
