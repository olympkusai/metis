from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from app.agent.state import AgentState
from app.tools import all_tools

SYSTEM_PROMPT = """Você é um analista especializado em investimentos em criptomoedas.
Use as ferramentas disponíveis para obter dados reais antes de responder.
Ferramentas disponíveis:
- get_live_price(symbol): preço atual
- get_indicators(symbol): variação, volume, máxima/mínima 24h
- calculate_risk(symbol, amount_usd): VaR, volatilidade e nível de risco
- search_market_news(query): notícias recentes
- get_top_cryptos(limit): ranking por volume

Dê uma análise clara e fundamentada nos dados obtidos."""

llm = ChatOpenAI(model="gpt-4o", temperature=0.2).bind_tools(all_tools)
tool_map = {t.name: t for t in all_tools}

def call_model(state: AgentState):
    """Nó LLM: decide qual tool chamar ou retorna resposta final."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    
    next_action = "tools" if response.tool_calls else "finalize"
    return {
        "messages": [response],
        "next_action": next_action,
        "intermediate_steps": state["intermediate_steps"],
        "final_answer": state["final_answer"],
        "user_id": state["user_id"]
    }

def tool_executor(state: AgentState):
    """Nó executor: executa as tools chamadas pelo LLM e registra resultados."""
    last_ai = state["messages"][-1]
    tool_messages = []
    steps = list(state["intermediate_steps"])

    for tool_call in last_ai.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        tool = tool_map.get(name)
        result = tool.invoke(args) if tool else f"Tool '{name}' não encontrada."
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        steps.append((name, str(result)))

    return {
        "messages": tool_messages,
        "next_action": "model",
        "intermediate_steps": steps,
        "final_answer": state["final_answer"],
        "user_id": state["user_id"]
    }

def finalize_node(state: AgentState):
    """Extrai a resposta final do último AIMessage sem tool_calls."""
    final = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final = msg.content
            break
    return {
        "messages": state["messages"],
        "next_action": "done",
        "intermediate_steps": state["intermediate_steps"],
        "final_answer": final,
        "user_id": state["user_id"]
    }

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("model", call_model)
    workflow.add_node("tools", tool_executor)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("model")
    workflow.add_conditional_edges(
        "model",
        lambda s: s["next_action"],
        {"tools": "tools", "finalize": "finalize"}
    )
    workflow.add_conditional_edges(
        "tools",
        lambda s: s["next_action"],
        {"model": "model"}
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()

def get_agent_graph():
    return build_agent_graph()
