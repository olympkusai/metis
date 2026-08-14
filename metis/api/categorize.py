"""Transaction categorization endpoint — uses GPT-4o to classify transactions
into the user's existing categories. Multilingual: works with any language."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Header
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from metis.config import get_settings
from metis.api.deps import get_jwt_verifier

logger = logging.getLogger(__name__)
router = APIRouter(tags=["categorization"])

# ── Schemas ────────────────────────────────────────────────────────────


class CategoryOption(BaseModel):
    id: str
    name: str


class CategorizeItem(BaseModel):
    transaction_id: str
    description: str = Field(min_length=1, max_length=500)
    amount: float = Field(gt=0)
    currency: str
    date: str = ""


class CategorizeRequest(BaseModel):
    transactions: list[CategorizeItem] = Field(min_length=1, max_length=50)
    categories: list[CategoryOption] = Field(min_length=1, max_length=100)


class CategorizeResultItem(BaseModel):
    transaction_id: str
    category_id: str
    category_name: str
    confidence: float


class CategorizeResponse(BaseModel):
    results: list[CategorizeResultItem]


# ── OpenAI client (singleton) ──────────────────────────────────────────

_openai_client: Optional[AsyncOpenAI] = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _openai_client


# ── Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a financial transaction categorization engine.
You receive a list of transactions and a list of available categories.
For each transaction, pick the single best-matching category from the list.

Rules:
- Return ONLY valid JSON. No markdown, no explanation.
- Use the category "id" field (not the name) in your response.
- If no category fits well, use the category whose name is closest to "Other" or "Outros".
- Confidence is a float between 0.0 and 1.0 — how certain you are the category fits.
- Consider the description, amount, and currency to infer the category.
- Descriptions may be in ANY language — you understand all of them.
- Merchant names, bank statement codes, and abbreviations are common — infer wisely.

Output format:
{"results": [
  {"transaction_id": "...", "category_id": "...", "category_name": "...", "confidence": 0.92},
  ...
]}
"""


def _build_user_prompt(transactions: list[CategorizeItem], categories: list[CategoryOption]) -> str:
    cat_lines = "\n".join(f'- id: "{c.id}", name: "{c.name}"' for c in categories)
    tx_lines = "\n".join(
        f'- transaction_id: "{t.transaction_id}", description: "{t.description}", '
        f"amount: {t.amount}, currency: {t.currency}"
        for t in transactions
    )
    return f"Available categories:\n{cat_lines}\n\nTransactions to categorize:\n{tx_lines}"


# ── Endpoint ───────────────────────────────────────────────────────────


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_transactions(
    req: CategorizeRequest,
    authorization: str = Header(...),
) -> CategorizeResponse:
    # Validate JWT — ensures only authenticated Pluto users can call this
    verifier = get_jwt_verifier()
    await verifier.verify(authorization.removeprefix("Bearer ").strip())

    client = _get_openai_client()

    # For small batches, send all in one call. For larger batches, split.
    BATCH = 25
    all_results: list[CategorizeResultItem] = []

    for i in range(0, len(req.transactions), BATCH):
        batch = req.transactions[i : i + BATCH]
        user_prompt = _build_user_prompt(batch, req.categories)

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)

            for item in parsed.get("results", []):
                all_results.append(
                    CategorizeResultItem(
                        transaction_id=item.get("transaction_id", ""),
                        category_id=item.get("category_id", ""),
                        category_name=item.get("category_name", ""),
                        confidence=float(item.get("confidence", 0.0)),
                    )
                )
        except Exception as e:
            logger.error("categorize: OpenAI call failed for batch %d: %s", i, e)
            # Return low-confidence "Other" for failed items
            other_cat = next((c for c in req.categories if c.name.lower() in ("other", "outros")), req.categories[-1])
            for t in batch:
                all_results.append(
                    CategorizeResultItem(
                        transaction_id=t.transaction_id,
                        category_id=other_cat.id,
                        category_name=other_cat.name,
                        confidence=0.0,
                    )
                )

    return CategorizeResponse(results=all_results)
