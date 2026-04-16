from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agent.graph import get_agent_graph, QuantAgentState

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent_graph()
    initial_state = QuantAgentState(
        messages=[HumanMessage(content=request.message)],
        user_id=request.user_id,
    )
    # recursion_limit = 8 nós × max 6 iterações internas + margem
    final_state = await agent.ainvoke(initial_state, config={"recursion_limit": 60})

    intermediate_steps = final_state.get("intermediate_steps", [])
    final_answer       = final_state.get("final_answer", "")

    # Formatar processo de raciocínio
    reasoning_steps = [
        {"step": i, "tool": tool_name, "result": result}
        for i, (tool_name, result) in enumerate(intermediate_steps, 1)
    ]

    # Metadados extras do pipeline multi-agente
    pipeline_meta = {
        "symbol":             final_state.get("symbol"),
        "timeframe":          final_state.get("timeframe"),
        "risk_level":         final_state.get("risk_level"),
        "signal_direction":   final_state.get("signal_direction"),
        "signal_confidence":  final_state.get("signal_confidence"),
        "gate_approved":      final_state.get("gate_approved"),
        "gate_reason":        final_state.get("gate_reason"),
        "anomalies_detected": final_state.get("anomalies_detected", []),
    }

    return {
        "response":    final_answer,
        "reasoning":   reasoning_steps,
        "tools_used":  [step[0] for step in intermediate_steps],
        "pipeline":    pipeline_meta,
    }
