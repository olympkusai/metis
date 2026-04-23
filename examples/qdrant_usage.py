"""Example usage of Qdrant for memory and RAG."""

import asyncio
from datetime import datetime
from app.memory.qdrant_memory import get_user_memory
from app.rag.qdrant_rag import get_news_rag


async def example_user_memory():
    """Example: Store and retrieve user memories."""
    memory = get_user_memory()
    
    user_id = "user_123"
    
    # Store memories
    await memory.store_memory(
        user_id=user_id,
        text="User prefers long-term BTC investments with 6-12 month horizon",
        metadata={
            "type": "preference",
            "timestamp": datetime.now().isoformat(),
        },
    )
    
    await memory.store_memory(
        user_id=user_id,
        text="User is risk-averse, max position size 5% per trade",
        metadata={
            "type": "risk_profile",
            "timestamp": datetime.now().isoformat(),
        },
    )
    
    # Retrieve relevant memories
    relevant = await memory.retrieve_memories(
        user_id=user_id,
        query="What is the user's risk tolerance?",
        top_k=2,
    )
    
    print("Relevant memories:")
    for mem in relevant:
        print(f"  - {mem['text'][:80]}... (score: {mem['score']:.3f})")
    
    # Get recent memories
    recent = await memory.get_recent_memories(user_id=user_id, limit=2)
    print("\nRecent memories:")
    for mem in recent:
        print(f"  - {mem['text'][:80]}...")


async def example_news_rag():
    """Example: Index and search crypto news."""
    rag = get_news_rag()
    
    # Index news articles
    news_items = [
        {
            "title": "Bitcoin breaks $70,000 resistance",
            "content": "Bitcoin surged past $70,000 for the first time, driven by institutional demand...",
            "url": "https://example.com/btc-70k",
            "timestamp": datetime.now().isoformat(),
            "source": "CryptoNews",
        },
        {
            "title": "Ethereum ETF approval expected soon",
            "content": "SEC is expected to approve spot Ethereum ETFs in the coming weeks...",
            "url": "https://example.com/eth-etf",
            "timestamp": datetime.now().isoformat(),
            "source": "CoinDesk",
        },
    ]
    
    point_ids = await rag.index_news(news_items)
    print(f"Indexed {len(point_ids)} news articles")
    
    # Search for relevant news
    results = await rag.search_news(
        query="Bitcoin price movement",
        top_k=2,
    )
    
    print("\nSearch results:")
    for result in results:
        print(f"  - {result['title'][:60]}... (score: {result['score']:.3f})")
    
    # Get recent news
    recent = await rag.get_recent_news(limit=2)
    print("\nRecent news:")
    for result in recent:
        print(f"  - {result['title'][:60]}...")


async def example_integration_with_agent():
    """Example: Integrate with LangGraph agent."""
    from app.storage.qdrant_client import get_qdrant_store
    from langchain_openai import OpenAIEmbeddings
    
    qdrant = get_qdrant_store()
    embeddings = OpenAIEmbeddings()
    
    # In your agent node, retrieve user context
    user_id = "user_123"
    query = "What is the user's investment strategy?"
    
    query_vector = embeddings.embed_query(query)
    
    results = await qdrant.search(
        collection_name="user_memories",
        query_vector=query_vector,
        limit=3,
        filter_payload={"user_id": user_id},
    )
    
    # Format for LLM context
    context = "\n".join([
        f"- {r.payload.get('text', '')}" for r in results
    ])
    
    print(f"\nUser context for agent:\n{context}")


async def main():
    """Run all examples."""
    print("=== User Memory Example ===")
    await example_user_memory()
    
    print("\n=== News RAG Example ===")
    await example_news_rag()
    
    print("\n=== Agent Integration Example ===")
    await example_integration_with_agent()


if __name__ == "__main__":
    asyncio.run(main())
