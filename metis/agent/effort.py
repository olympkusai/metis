"""Effort configuration for the agentic runtime.

Three effort levels control the trade-off between speed and capability:

- low:    gpt-4o-mini, 5 iterations, 4-msg window, compressed prompt,
          no analytical frameworks. ~5-10s latency. Best for simple
          queries ("quanto gastei hoje?", "qual meu saldo?").

- medium: gpt-4o, 10 iterations, 8-msg window, full prompt.
          ~15-25s latency. Best for multi-step queries ("cria uma
          transação de 50 no mercado", "compara meus gastos com
          mês passado").

- high:   gpt-4o, 15 iterations, 20-msg window, full prompt +
          analytical frameworks, pre-fetch all reports. ~30-60s
          latency. Best for complex analysis ("analisa minha saúde
          financeira e sugere melhorias").

Auto-selection picks the level based on the user's message:
- Simple lookups / greetings / single action → low
- Multi-tool queries / comparisons / transactions → medium
- Analysis / planning / "analisa" / "sugere" / multi-step → high
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EffortLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


@dataclass(frozen=True)
class EffortConfig:
    """Runtime parameters for a given effort level."""
    level: str
    model: str
    max_iterations: int
    history_limit: int
    temperature: float
    # Pre-fetch reports in graph_v2 (reduces ReAct iterations)
    prefetch_spending: bool
    prefetch_cashflow: bool
    prefetch_budget: bool
    prefetch_goals: bool
    # Use compressed prompt (shorter, no frameworks)
    compressed_prompt: bool
    # Include analytical frameworks in the prompt
    include_frameworks: bool

    @property
    def is_low(self) -> bool:
        return self.level == EffortLevel.LOW.value


# ── Presets ──

_LOW = EffortConfig(
    level="low",
    model="gpt-4o-mini",
    max_iterations=5,
    history_limit=4,
    temperature=0.0,
    prefetch_spending=True,
    prefetch_cashflow=True,
    prefetch_budget=False,
    prefetch_goals=False,
    compressed_prompt=True,
    include_frameworks=False,
)

_MEDIUM = EffortConfig(
    level="medium",
    model="gpt-4o",
    max_iterations=10,
    history_limit=8,
    temperature=0.1,
    prefetch_spending=True,
    prefetch_cashflow=True,
    prefetch_budget=False,
    prefetch_goals=False,
    compressed_prompt=False,
    include_frameworks=False,
)

_HIGH = EffortConfig(
    level="high",
    model="gpt-4o",
    max_iterations=15,
    history_limit=20,
    temperature=0.2,
    prefetch_spending=True,
    prefetch_cashflow=True,
    prefetch_budget=True,
    prefetch_goals=True,
    compressed_prompt=False,
    include_frameworks=True,
)

_PRESETS: dict[str, EffortConfig] = {
    EffortLevel.LOW.value: _LOW,
    EffortLevel.MEDIUM.value: _MEDIUM,
    EffortLevel.HIGH.value: _HIGH,
}


def get_effort_config(level: str = "auto", user_message: str = "") -> EffortConfig:
    """Get the effort config for a given level.

    If level is "auto", selects based on the user's message.
    """
    if level == EffortLevel.AUTO.value:
        return _auto_select(user_message)
    return _PRESETS.get(level, _MEDIUM)


# ── Auto-selection ──

# High-effort keywords: analysis, planning, multi-step reasoning
# Note: no trailing \b — many keywords are stems (analis, recomend, planej)
# that need to match word continuations (analisa, recomenda, planejamento)
_HIGH_KEYWORDS = re.compile(
    r"\b(analis|sugir|sugest|recomend|planej|planeja|estrat|saúde|saude|melhor|"
    r"otimiz|investig|resumo completo|visão geral|"
    r"diagnóstico|diagnostico|planejamento|orçamento ideal|"
    r"dicas|conselho|orientação|orientacao|sair.*dívid|sair.*divid)",
    re.IGNORECASE,
)

# Low-effort keywords: simple lookups, greetings, single actions
_LOW_KEYWORDS = re.compile(
    r"\b(olá|ola|oi|bom dia|boa tarde|boa noite|obrigad|valeu|"
    r"saldo|quanto.*tem|quanto.*gastei|quanto.*recebi|"
    r"minha.*conta|meus.*gastos|transações.*de|transacoes.*de|"
    r"quanto.*gast|qual.*valor|mostra|ver|listar|"
    r"criar?|registrar?|gastei|recebi|paguei|"
    r"sim|não|nao|confirmar|cancelar)",
    re.IGNORECASE,
)

# Medium-effort: multiple tools, comparisons, date ranges
_MEDIUM_KEYWORDS = re.compile(
    r"\b(último.*mês|ultimo.*mes|mês.*passado|mes.*passado|"
    r"comparar|compar|versus|vs|diferença|diferenca|"
    r"orçamento|orcamento|meta|metas|"
    r"parcela|parcelamento|recorr|"
    r"transfer|invest|poup)",
    re.IGNORECASE,
)


def _auto_select(message: str) -> EffortConfig:
    """Select effort level based on the user's message.

    Heuristics (checked in order of priority):
    1. High: analysis/planning/suggestions keywords
    2. Medium: comparisons, multi-tool queries, date ranges
    3. Low: greetings, simple lookups, confirmations, single actions
    4. Medium: everything else (default)
    """
    if not message or not message.strip():
        return _MEDIUM

    msg = message.strip()

    # High: analysis / planning / multi-step
    if _HIGH_KEYWORDS.search(msg):
        return _HIGH

    # Medium: comparisons, date ranges, multi-tool queries
    if _MEDIUM_KEYWORDS.search(msg):
        return _MEDIUM

    # Low: greetings, simple lookups, confirmations, single actions
    # Also: short messages (< 20 chars) that aren't questions
    if _LOW_KEYWORDS.search(msg) or (len(msg) < 20 and "?" not in msg):
        return _LOW

    # Medium: default
    return _MEDIUM
