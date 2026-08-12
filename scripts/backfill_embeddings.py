"""Backfill embeddings for existing chat messages.

Run via railway ssh:
    railway ssh --service metis -- python3 scripts/backfill_embeddings.py

Embeds all messages where embedding IS NULL using text-embedding-3-small.
"""
import asyncio
import asyncpg
import os
from openai import AsyncOpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_INPUT_CHARS = 8000
BATCH_SIZE = 50


async def backfill():
    url = os.getenv("CONVERSATION_DATABASE_URL")
    if not url:
        print("ERROR: CONVERSATION_DATABASE_URL not set")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    conn = await asyncpg.connect(url)
    client = AsyncOpenAI(api_key=api_key)

    # Count messages without embeddings
    total = await conn.fetchval(
        "SELECT count(*) FROM chat_messages WHERE embedding IS NULL"
    )
    print(f"Found {total} messages without embeddings")

    if total == 0:
        print("Nothing to backfill — all messages already have embeddings.")
        await conn.close()
        return

    done = 0
    failed = 0

    while True:
        rows = await conn.fetch(
            "SELECT id, content FROM chat_messages "
            "WHERE embedding IS NULL ORDER BY created_at "
            "LIMIT $1",
            BATCH_SIZE,
        )
        if not rows:
            break

        # Batch embed (OpenAI supports up to 2048 inputs per request)
        texts = [r["content"][:MAX_INPUT_CHARS] for r in rows]
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
        except Exception as e:
            print(f"  Batch embed failed: {e}, retrying one by one...")
            for r in rows:
                try:
                    resp = await client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=r["content"][:MAX_INPUT_CHARS],
                    )
                    emb = resp.data[0].embedding
                    await conn.execute(
                        "UPDATE chat_messages SET embedding = $1 WHERE id = $2",
                        emb, r["id"],
                    )
                    done += 1
                except Exception as e2:
                    print(f"    Failed for {r['id']}: {e2}")
                    failed += 1
            continue

        # Store embeddings
        for r, data in zip(rows, response.data):
            try:
                await conn.execute(
                    "UPDATE chat_messages SET embedding = $1 WHERE id = $2",
                    data.embedding, r["id"],
                )
                done += 1
            except Exception as e:
                print(f"  Failed to store {r['id']}: {e}")
                failed += 1

        print(f"  Progress: {done}/{total} ({failed} failed)")

    print(f"\nBackfill complete: {done} embedded, {failed} failed, {total} total")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(backfill())
