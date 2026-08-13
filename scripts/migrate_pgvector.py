import asyncio, asyncpg, os
async def run():
    url = os.getenv("CONVERSATION_DATABASE_URL")
    conn = await asyncpg.connect(url)
    print("Enabling pgvector...")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("Altering embedding column...")
    try:
        await conn.execute("ALTER TABLE chat_messages ALTER COLUMN embedding TYPE vector(1536) USING embedding::float8[]::vector(1536)")
        print("Column altered successfully")
    except Exception as e:
        print("Alter skipped: " + str(e))
    print("Creating index...")
    try:
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding ON chat_messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100) WHERE embedding IS NOT NULL")
        print("Index created")
    except Exception as e:
        print("Index skipped: " + str(e))
    rows = await conn.fetch("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ver = rows[0]["extversion"] if rows else "NO"
    print("pgvector installed: " + str(ver))
    rows = await conn.fetch("SELECT udt_name FROM information_schema.columns WHERE table_name = 'chat_messages' AND column_name = 'embedding'")
    udt = rows[0]["udt_name"] if rows else "NOT FOUND"
    print("embedding column type: " + str(udt))
    count = await conn.fetchval("SELECT count(*) FROM chat_messages WHERE embedding IS NULL")
    print("Messages without embedding: " + str(count))
    await conn.close()
asyncio.run(run())
