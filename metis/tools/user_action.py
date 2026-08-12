"""User action tool — lets the LLM request structured user input.

When the LLM needs confirmation, a choice, or approval before executing
an action, it calls request_user_action instead of asking in free text.
The runtime intercepts this tool call and emits an SSE action_request
event with structured options. The frontend renders buttons/cards.

The tool returns a placeholder string so the ReAct loop can continue
(the actual user response arrives in the next conversation turn).
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def request_user_action(
    title: str,
    message: str,
    options: list[str],
    *,
    action_type: str = "confirm",
    danger: bool = False,
) -> str:
    """Solicita uma ação do usuário via interface estruturada (botões/cards).

    Use esta ferramenta SEMPRE que precisar de uma decisão do usuário antes
    de executar algo. NÃO pergunte em texto livre — use esta tool para que
    o frontend renderize botões clicáveis.

    Casos de uso:
    - Confirmar transação: "Criar despesa de R$ 50 no mercado?" → ["Confirmar", "Cancelar"]
    - Escolher conta: "Qual conta?" → ["Nubank", "Itaú", "Carteira"]
    - Confirmar exclusão (danger=True): "Excluir meta?" → ["Excluir", "Manter"]
    - Escolher data: "Qual data?" → ["Hoje", "Ontem", "Outra"]

    Args:
        title: Título curto da ação (ex: "Confirmar transação").
        message: Descrição detalhada do que será feito (ex: "Criar despesa
                 de R$ 50 no mercado com data de hoje.").
        options: Lista de opções clicáveis (ex: ["Confirmar", "Cancelar"]).
                 A primeira opção deve ser a ação principal/afirmativa.
        action_type: Tipo da ação — "confirm" (sim/não), "select" (escolher
                     uma opção), "input" (texto livre). Padrão: "confirm".
        danger: Se True, destaca a ação como destrutiva/perigosa (ex: excluir).
                Padrão: False.

    Returns:
        String indicando que a ação está pendente. O usuário responderá
        no próximo turno da conversa.
    """
    return (
        f"[AÇÃO PENDENTE] {title}: {message}\n"
        f"Opções: {', '.join(options)}\n"
        f"Aguardando resposta do usuário no próximo turno."
    )
