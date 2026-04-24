"""Run Qdrant migration to create collections."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.storage.qdrant_client import get_qdrant_store

async def main():
    print("Running Qdrant migration to create collections...")
    
    qdrant_store = get_qdrant_store()
    await qdrant_store.ensure_collections()
    
    print("✓ Collections created/verified:")
    print("  - user_memories (1536 dims, COSINE)")
    print("  - crypto_news (1536 dims, COSINE)")
    print("  - conversation_history (1536 dims, COSINE)")
    print("\nMigration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
