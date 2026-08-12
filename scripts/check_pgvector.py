import asyncio, asyncpg, os
async def check():
    url = os.getenv("CONVERSATION_DATABASE_URL")
    conn = await asyncpg.connect(url)
    rows = await conn.fetch("SELECT * FROM pg_available_extensions WHERE name = 'vector'")
    if rows:
        print(f"pgvector AVAILABLE: version {rows[0]['default_version']}")
    else:
        print("pgvector NOT available")
    rows = await conn.fetch("SELECT * FROM pg_extension WHERE extname = 'vector'")
    if rows:
        print(f"pgvector INSTALLED: version {rows[0]['extversion']}")
    else:
        print("pgvector not yet installed")
    version = await conn.fetchval("SELECT version()")
    print(f"Postgres: {version[:80]}")
    rows = await conn.fetch("SELECT column_name, data_type, udt_name FROM information_schema.columns WHERE table_name = 'chat_messages' AND column_name = 'embedding'")
    if rows:
        print(f"embedding: type={rows[0]['data_type']}, udt={rows[0]['udt_name']}")
    else:
        print("embedding column not found")
    count = await conn.fetchval("SELECT count(*) FROM chat_messages")
    print(f"Total messages: {count}")
    await conn.close()
asyncio.run(check())
