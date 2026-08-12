"""Finance tools — on-demand Pluto report lookups for the finance persona.

These wrap PlutoApiClient report endpoints as LangChain tools the reasoning
LLM calls when a question needs more than the always-fetched financial
profile/accounts (see finance_context_node in app/agent/graph.py).

Auth: the user's bearer token is never an LLM-visible tool argument (it would
leak into the tool-call trace, and the LLM has no legitimate way to "know"
it). It's threaded via a `contextvars.ContextVar`, set once per request by
finance_context_node before the reasoning node runs, and read here — a
ContextVar (not a plain module global) because concurrent requests from
different users must not see each other's token.
"""

from __future__ import annotations

import json
from contextvars import ContextVar

from langchain_core.tools import tool

from metis.pluto_client import PlutoApiError, get_pluto_client

_auth_token: ContextVar[str] = ContextVar("finance_auth_token", default="")


def set_auth_token(token: str) -> None:
    """Sets the bearer token finance tools use for the current request."""
    _auth_token.set(token)


def _current_token() -> str:
    token = _auth_token.get()
    if not token:
        raise RuntimeError("finance auth token not set for this request")
    return token


@tool
async def get_spending_by_category(date_from: str = "", date_to: str = "") -> str:
    """Retorna os gastos do usuário agrupados por categoria no período informado
    (formato YYYY-MM-DD). Se as datas forem omitidas, usa o mês corrente."""
    try:
        data = await get_pluto_client().spending_by_category(
            token=_current_token(), date_from=date_from, date_to=date_to
        )
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


@tool
async def get_cashflow(date_from: str = "", date_to: str = "") -> str:
    """Retorna o fluxo de caixa (receitas x despesas por mês) do usuário no
    período informado (formato YYYY-MM-DD). Se omitido, usa o mês corrente."""
    try:
        data = await get_pluto_client().cashflow(
            token=_current_token(), date_from=date_from, date_to=date_to
        )
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


@tool
async def get_budget_progress() -> str:
    """Retorna o progresso de cada orçamento ativo do usuário no período
    corrente: quanto já foi gasto vs. o valor orçado por categoria."""
    try:
        data = await get_pluto_client().budget_progress(token=_current_token())
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


@tool
async def get_goal_summary() -> str:
    """Retorna um resumo consolidado de todas as metas financeiras do usuário:
    percentual concluído, dias restantes até a data-alvo, prioridade."""
    try:
        data = await get_pluto_client().goal_summary(token=_current_token())
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


@tool
async def get_recurrences_due(within_days: int = 7) -> str:
    """Retorna as contas recorrentes do usuário que vencem nos próximos
    `within_days` dias (padrão 7)."""
    try:
        data = await get_pluto_client().recurrences_due(
            token=_current_token(), within_days=within_days
        )
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


@tool
async def list_transactions_filtered(
    category_id: str = "",
    type: str = "",
    date_from: str = "",
    date_to: str = "",
    page: int = 1,
) -> str:
    """Lista transações do usuário com filtros opcionais por categoria, tipo
    (expense/income/saving/investment/dividend/investment_withdrawal/transfer)
    e período (YYYY-MM-DD). Use quando precisar ver transações individuais,
    não apenas totais agregados.

    category_id pode ser o UUID da categoria OU o nome da categoria
    (ex: "mercado", "lazer", "transporte"). Se for um nome, a tool
    resolve automaticamente para o UUID correspondente."""
    try:
        client = get_pluto_client()
        token = _current_token()

        # Resolve category name → UUID if needed
        resolved_category_id = category_id
        if category_id and not _is_uuid(category_id):
            try:
                cats_data = await client.list_categories(token=token)
                categories = (
                    cats_data.get("categories", cats_data)
                    if isinstance(cats_data, dict)
                    else cats_data
                )
                if isinstance(categories, list):
                    cat_lower = category_id.lower().strip()
                    for cat in categories:
                        cat_name = ""
                        cat_uuid = ""
                        if isinstance(cat, dict):
                            cat_name = str(cat.get("name", "")).lower()
                            cat_uuid = cat.get("id", cat.get("category_id", ""))
                        elif isinstance(cat, str):
                            cat_name = cat.lower()
                        if cat_name == cat_lower and cat_uuid:
                            resolved_category_id = cat_uuid
                            break
            except PlutoApiError:
                # If category lookup fails, try with the original value
                pass

        data = await client.list_transactions(
            token=token,
            category_id=resolved_category_id,
            type=type,
            date_from=date_from,
            date_to=date_to,
            page=page,
        )
        return json.dumps(data)
    except PlutoApiError as e:
        return json.dumps({"error": str(e)})


def _is_uuid(value: str) -> bool:
    """Check if a string looks like a UUID."""
    if not value or len(value) < 32:
        return False
    try:
        import uuid as _uuid
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


finance_tools = [
    get_spending_by_category,
    get_cashflow,
    get_budget_progress,
    get_goal_summary,
    get_recurrences_due,
    list_transactions_filtered,
]
