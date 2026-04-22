from pathlib import Path
import sys

import uvicorn
from fastapi import FastAPI

if __package__ in {None, ""}:
    # Support direct execution via `python app/main.py` by ensuring the
    # project root is searched before any installed `app` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import register_routes

app = FastAPI(title="Crypto Agent")

register_routes(app)

@app.get("/")
async def root():
    return {"message": "Crypto Agent API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
