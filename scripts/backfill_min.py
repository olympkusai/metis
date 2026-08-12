import asyncio, asyncpg, os
from openai import AsyncOpenAI

async def main():
    url = os.getenv('CONVERSATION_DATABASE_URL')
    key = os.getenv('OPENAI_API_KEY')
    conn = await asyncpg.connect(url)
    client = AsyncOpenAI(api_key=key)
    total = await conn.fetchval('SELECT count(*) FROM chat_messages WHERE embedding IS NULL')
    print('Backfilling ' + str(total) + ' messages...')
    done = 0
    while True:
        rows = await conn.fetch("SELECT id, content FROM chat_messages WHERE embedding IS NULL ORDER BY created_at LIMIT 50")
        if not rows:
            break
        texts = [r['content'][:8000] for r in rows]
        resp = await client.embeddings.create(model='text-embedding-3-small', input=texts)
        for r, d in zip(rows, resp.data):
            await conn.execute('UPDATE chat_messages SET embedding = $1 WHERE id = $2', d.embedding, r['id'])
            done += 1
        print('  ' + str(done) + '/' + str(total))
    print('Done: ' + str(done) + ' embedded')
    await conn.close()

asyncio.run(main())
