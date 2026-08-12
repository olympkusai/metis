"""Context manager for the agent runtime.

Manages the LLM context window to prevent overflow in long conversations
and tool-heavy executions. Three main responsibilities:

1. Token counting — precise per-model token counts via tiktoken
2. Tool result truncation — cap each tool result to a token budget
3. History summarization — when history exceeds budget, summarize old
   messages into a compact block via gpt-4o-mini

Budget allocation (for gpt-4o's 128K window):
  - system_prompt:  ~6K tokens (fixed)
  - tools_schema:   ~3K tokens (fixed, from bind_tools)
  - extra_context:  ~4K tokens (profile, accounts, categories, prefetch)
  - history:        ~8K tokens (managed — summarize when exceeded)
  - tool_results:   ~6K tokens (managed — truncate each result)
  - output_budget:  ~2K tokens (reserved for LLM response)
  - total:          ~29K tokens used, ~99K headroom

For gpt-4o-mini (low effort), the window is also 128K but we use a
much smaller budget since queries are simpler:
  - history: ~2K tokens (4 messages)
  - tool_results: ~3K tokens
"""
from __future__ import annotations

import json
import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# ── Model context windows ──
_MODEL_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_000,
}

# ── Budget presets (tokens) ──
_BUDGETS: dict[str, dict[str, int]] = {
    "low": {
        "system_prompt": 2_000,    # compressed prompt
        "tools_schema": 3_000,
        "extra_context": 3_000,
        "history": 2_000,
        "tool_results": 3_000,     # per-result max
        "output_budget": 1_500,
    },
    "medium": {
        "system_prompt": 7_000,    # full prompt (~6.2K)
        "tools_schema": 3_000,
        "extra_context": 4_000,
        "history": 8_000,
        "tool_results": 4_000,     # per-result max
        "output_budget": 2_000,
    },
    "high": {
        "system_prompt": 7_000,
        "tools_schema": 3_000,
        "extra_context": 8_000,    # more prefetch data
        "history": 20_000,
        "tool_results": 6_000,     # per-result max
        "output_budget": 4_000,
    },
}

# ── Token encoders (cached) ──
_encoders: dict[str, tiktoken.Encoding] = {}


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get a tiktoken encoder for the model (cached)."""
    if model not in _encoders:
        try:
            _encoders[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base (used by gpt-4o family)
            _encoders[model] = tiktoken.get_encoding("cl100k_base")
    return _encoders[model]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string using the model's tokenizer."""
    if not text:
        return 0
    return len(_get_encoder(model).encode(text))


def count_message_tokens(message: Any, model: str = "gpt-4o") -> int:
    """Count tokens in a LangChain message object or dict."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "") or ""
    return count_tokens(content, model)


class ContextManager:
    """Manages context window budget for the agent runtime.

    Usage:
        cm = ContextManager(model="gpt-4o", effort="medium")
        truncated = cm.truncate_tool_result(result_str)
        history_msgs = cm.summarize_if_needed(messages, llm_mini)
    """

    def __init__(self, model: str = "gpt-4o", effort: str = "medium"):
        self.model = model
        self.effort = effort
        self.window = _MODEL_WINDOWS.get(model, 128_000)
        self.budget = _BUDGETS.get(effort, _BUDGETS["medium"])

    @property
    def max_tool_result_tokens(self) -> int:
        """Max tokens per tool result."""
        return self.budget["tool_results"]

    @property
    def max_history_tokens(self) -> int:
        """Max tokens for conversation history."""
        return self.budget["history"]

    @property
    def total_budget(self) -> int:
        """Total token budget (all components)."""
        return sum(self.budget.values())

    @property
    def available_for_history(self) -> int:
        """Tokens available for history after fixed costs."""
        fixed = (
            self.budget["system_prompt"]
            + self.budget["tools_schema"]
            + self.budget["extra_context"]
            + self.budget["output_budget"]
        )
        return self.window - fixed

    def truncate_tool_result(self, result: str) -> str:
        """Truncate a tool result to fit within the token budget.

        Preserves the beginning and end of the result, with a notice
        in the middle showing how much was omitted.
        """
        if not result:
            return result

        token_count = count_tokens(result, self.model)
        max_tokens = self.max_tool_result_tokens

        if token_count <= max_tokens:
            return result

        # Encode, keep first 60% and last 30%, leave 10% for the notice
        encoder = _get_encoder(self.model)
        tokens = encoder.encode(result)

        keep_start = int(max_tokens * 0.6)
        keep_end = int(max_tokens * 0.3)
        omitted = token_count - keep_start - keep_end

        start_text = encoder.decode(tokens[:keep_start])
        end_text = encoder.decode(tokens[-keep_end:])
        notice = (
            f"\n\n[... {omitted} tokens omitidos "
            f"({omitted * 100 // token_count}% do resultado) ...]\n\n"
        )

        return start_text + notice + end_text

    def count_history_tokens(self, messages: list) -> int:
        """Count total tokens in a list of messages."""
        return sum(count_message_tokens(m, self.model) for m in messages)

    def should_summarize(self, messages: list) -> bool:
        """Check if history exceeds the budget and should be summarized."""
        history_tokens = self.count_history_tokens(messages)
        threshold = min(
            self.max_history_tokens,
            int(self.available_for_history * 0.8),
        )
        return history_tokens > threshold

    def select_messages_to_summarize(self, messages: list) -> tuple[list, list]:
        """Split messages into (to_summarize, to_keep).

        Keeps the most recent messages (up to budget) and marks older
        ones for summarization. Always keeps at least the last 2 messages
        (the current user message + the previous turn).
        """
        if len(messages) <= 2:
            return [], messages

        # Walk from the end, keeping messages until we hit the budget
        kept_tokens = 0
        kept_count = 0
        for m in reversed(messages):
            msg_tokens = count_message_tokens(m, self.model)
            if kept_tokens + msg_tokens > self.max_history_tokens and kept_count >= 2:
                break
            kept_tokens += msg_tokens
            kept_count += 1

        split_idx = len(messages) - kept_count
        if split_idx <= 0:
            return [], messages

        return messages[:split_idx], messages[split_idx:]

    async def summarize_messages(
        self,
        messages: list,
        llm: Any,
    ) -> str:
        """Summarize old messages into a compact context block.

        Uses gpt-4o-mini (passed as llm) to generate a concise summary
        that preserves key facts: amounts, dates, actions taken, user
        preferences.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build the conversation text to summarize
        conv_text = ""
        for m in messages:
            if isinstance(m, dict):
                role = m.get("type", m.get("role", "unknown"))
                content = m.get("content", "")
            else:
                role = "user" if isinstance(m, HumanMessage) else "assistant"
                content = getattr(m, "content", "") or ""
            conv_text += f"{role}: {content}\n"

        system_prompt = (
            "Você é um sumarizador de conversas financeiras. "
            "Crie um resumo CONCISO (máximo 300 tokens) que preserva:\n"
            "- Valores monetários mencionados\n"
            "- Ações realizadas (transações criadas, metas, etc)\n"
            "- Datas importantes\n"
            "- Preferências do usuário\n"
            "- Perguntas pendentes ou não respondidas\n"
            "NÃO inclua saudações ou conversa fiada. Apenas fatos.\n"
            "Formato: bullet points em português."
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Resuma esta conversa:\n\n{conv_text}"),
            ])
            summary = response.content.strip()
            logger.info(
                f"[context_manager] Summarized {len(messages)} messages "
                f"({self.count_history_tokens(messages)} tokens) "
                f"into {count_tokens(summary, self.model)} tokens"
            )
            return f"[RESUMO DA CONVERSA ANTERIOR]\n{summary}"
        except Exception as e:
            logger.warning(f"[context_manager] Summarization failed: {e}")
            # Fall back to simple truncation (keep first 500 chars of each)
            parts = []
            for m in messages:
                content = getattr(m, "content", "") or (m.get("content", "") if isinstance(m, dict) else "")
                if content:
                    parts.append(content[:200])
            return f"[RESUMO DA CONVERSA ANTERIOR]\n" + "\n".join(parts)

    def truncate_reasoning_trail(
        self,
        trail: list[tuple[str, str, str]],
        max_entries: int = 5,
    ) -> list[tuple[str, str, str]]:
        """Keep only the last N reasoning trail entries.

        The trail grows with each node execution. In a single-node graph
        (v2), it's usually 1 entry, but in multi-turn conversations it
        can accumulate. Cap at max_entries to prevent unbounded growth.
        """
        if len(trail) <= max_entries:
            return trail
        return trail[-max_entries:]
