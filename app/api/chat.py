from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import get_agent_graph, get_finance_agent_graph, QuantAgentState
from app.memory.conversation_history import (
    get_conversation_history,
    MessageRole,
)
import json
import asyncio
import uuid

router = APIRouter()


def _as_dict(obj) -> dict:
    """Normalize AssetState / BaseModel / None into a plain dict.

    LangGraph stream updates carry `assets: dict[str, AssetState]`, so the
    values are Pydantic objects (not dicts). This helper lets the SSE layer
    use `.get(...)` uniformly without leaking Pydantic into the API code.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    domain: Literal["finance", "crypto"] = "finance"


def _extract_bearer_token(authorization: Optional[str]) -> str:
    """Extracts the raw JWT from an `Authorization: Bearer <token>` header.

    Returns "" when absent — callers (finance_context_node) treat a missing
    token as an unauthenticated request and fail gracefully rather than crash.
    """
    if not authorization:
        return ""
    return authorization.removeprefix("Bearer ").strip()


@router.post("/chat")
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    auth_token = _extract_bearer_token(authorization)
    # Generate session_id if not provided
    session_id = request.session_id or f"{request.user_id}:{uuid.uuid4().hex}"
    
    # Save user message
    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=request.user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )
    
    # Load conversation history (20 messages from session + 10 global context)
    session_history = await conv_history.get_conversation_history(
        user_id=request.user_id,
        session_id=session_id,
        limit=20,
    )
    global_context = await conv_history.get_global_context(
        user_id=request.user_id,
        limit=10,
        exclude_session_id=session_id,
    )
    
    # Build messages list with history
    messages = []
    
    # Add global context as system context
    if global_context:
        context_summary = "Previous conversations context:\n"
        for msg in global_context[-5:]:  # Last 5 global messages
            context_summary += f"{msg['role']}: {msg['content'][:200]}...\n"
        messages.append(HumanMessage(content=context_summary))
    
    # Add session history
    for msg in session_history[:-1]:  # All except the one we just saved
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content']))
    
    # Add current message
    messages.append(HumanMessage(content=request.message))
    
    agent = get_agent_graph() if request.domain == "crypto" else get_finance_agent_graph()
    initial_state = QuantAgentState(
        messages=messages,
        user_id=request.user_id,
        auth_token=auth_token,
    )
    # recursion_limit = 8 nós × max 6 iterações internas + margem
    final_state = await agent.ainvoke(initial_state, config={"recursion_limit": 60})

    intermediate_steps = final_state.get("intermediate_steps_global", [])
    final_answer       = final_state.get("final_answer", "")
    cot                 = final_state.get("cot", "")

    # Formatar processo de raciocínio
    reasoning_steps = [
        {"step": i, "tool": tool_name, "result": result}
        for i, (tool_name, result) in enumerate(intermediate_steps, 1)
    ]

    # ── Multi-asset view: campos por-ativo agora vivem em `assets[symbol]`. ──
    primary_symbol = final_state.get("primary_symbol") or ""
    all_symbols = final_state.get("symbols", []) or ([primary_symbol] if primary_symbol else [])
    assets_dict = final_state.get("assets", {}) or {}
    primary = assets_dict.get(primary_symbol, {}) or {}

    # Metadados extras do pipeline multi-agente
    pipeline_meta = {
        "symbol":             primary_symbol,
        "symbols":            all_symbols,
        "timeframe":          final_state.get("timeframe"),
        "risk_level":         primary.get("risk_level"),
        "signal_direction":   primary.get("signal_direction"),
        "signal_confidence":  primary.get("signal_confidence"),
        "gate_approved":      final_state.get("gate_approved"),
        "gate_reason":        final_state.get("gate_reason"),
        "anomalies_detected": primary.get("anomalies_detected", []),
        "forecast": {
            "status": primary.get("forecast_status"),
            "direction": primary.get("forecast_direction"),
            "confidence": primary.get("forecast_confidence"),
            "predicted_price": primary.get("forecast_predicted_price"),
            "return_pct": primary.get("forecast_return_pct"),
            "actionable": primary.get("forecast_actionable"),
            "warnings": primary.get("forecast_warnings", []),
        },
        # Quando há comparação, exponha um snapshot compacto dos comparativos
        # para o frontend renderizar lado a lado se quiser.
        "compared": [
            {
                "symbol": s,
                "live_price": (assets_dict.get(s) or {}).get("live_price"),
                "price_change_pct": (assets_dict.get(s) or {}).get("price_change_pct"),
                "risk_level": (assets_dict.get(s) or {}).get("risk_level"),
                "sharpe": (assets_dict.get(s) or {}).get("sharpe"),
                "max_drawdown": (assets_dict.get(s) or {}).get("max_drawdown"),
                "volatility_annualized": (assets_dict.get(s) or {}).get("volatility_annualized"),
                "macro_regime": (assets_dict.get(s) or {}).get("macro_regime"),
                "forecast_return_pct": (assets_dict.get(s) or {}).get("forecast_return_pct"),
                "forecast_confidence": (assets_dict.get(s) or {}).get("forecast_confidence"),
            }
            for s in all_symbols if s and s != primary_symbol
        ],
    }
    
    # Save assistant response with complete metadata
    await conv_history.save_message(
        user_id=request.user_id,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=final_answer,
        metadata={
            "pipeline_summary": pipeline_meta,
            "reasoning_steps": reasoning_steps,
            "tools_used": [step[0] for step in intermediate_steps],
            "chain_of_thought": cot,
        },
    )

    return {
        "response":    final_answer,
        "reasoning":   reasoning_steps,
        "tools_used":  [step[0] for step in intermediate_steps],
        "pipeline":    pipeline_meta,
        "thought":     cot,
        "session_id":  session_id,
    }


@router.post("/streaming/chat")
async def streaming_chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    """Streaming endpoint that emits each node execution as an SSE event."""
    auth_token = _extract_bearer_token(authorization)

    # Generate session_id if not provided
    session_id = request.session_id or f"{request.user_id}:{uuid.uuid4().hex}"
    
    # Save user message
    conv_history = get_conversation_history()
    await conv_history.save_message(
        user_id=request.user_id,
        session_id=session_id,
        role=MessageRole.USER,
        content=request.message,
    )
    
    async def event_generator():
        # Load conversation history (20 messages from session + 10 global context)
        session_history = await conv_history.get_conversation_history(
            user_id=request.user_id,
            session_id=session_id,
            limit=20,
        )
        global_context = await conv_history.get_global_context(
            user_id=request.user_id,
            limit=10,
            exclude_session_id=session_id,
        )
        
        # Build messages list with history
        messages = []
        
        # Add global context as system context
        if global_context:
            context_summary = "Previous conversations context:\n"
            for msg in global_context[-5:]:  # Last 5 global messages
                context_summary += f"{msg['role']}: {msg['content'][:200]}...\n"
            messages.append(HumanMessage(content=context_summary))
        
        # Add session history
        for msg in session_history[:-1]:  # All except the one we just saved
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        # Add current message
        messages.append(HumanMessage(content=request.message))
        
        agent = get_agent_graph() if request.domain == "crypto" else get_finance_agent_graph()
        initial_state = QuantAgentState(
            messages=messages,
            user_id=request.user_id,
            auth_token=auth_token,
        )

        # Accumulate state updates to build final state
        accumulated_state = {}
        
        # Stream node executions with reasoning
        async for event in agent.astream(
            initial_state,
            config={"recursion_limit": 60},
            stream_mode="updates"
        ):
            # event is a dict of node_name -> state_update
            for node_name, state_update in event.items():
                # Accumulate state
                accumulated_state.update(state_update)
                
                # Extract relevant information from each node
                step_data = {
                    "node": node_name,
                    "type": "node_execution",
                    "timestamp": asyncio.get_event_loop().time(),
                    "reasoning": "",
                    "thought": ""
                }
                
                # Extract CoT if available (chain-of-thought)
                if "cot" in state_update and state_update["cot"]:
                    step_data["thought"] = state_update["cot"]
                
                # Extract reasoning from messages if available
                if "messages" in state_update:
                    messages = state_update["messages"]
                    if messages:
                        # Get the last AIMessage which contains the LLM's reasoning
                        last_message = messages[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            # Extract content - handle both string and dict formats
                            content = last_message.content
                            if isinstance(content, dict) and "content" in content:
                                content = content["content"]
                            elif isinstance(content, str):
                                try:
                                    # Try to parse if it's a JSON string
                                    import json
                                    parsed = json.loads(content)
                                    if isinstance(parsed, dict) and "content" in parsed:
                                        content = parsed["content"]
                                except:
                                    pass  # Keep as-is if not valid JSON
                            
                            # Truncate long reasoning to keep it readable
                            if isinstance(content, str) and len(content) > 500:
                                content = content[:500] + "..."
                            step_data["reasoning"] = content
                
                # ── Helper: extrai snapshot do primary asset do update ──
                primary_sym = (
                    state_update.get("primary_symbol")
                    or accumulated_state.get("primary_symbol")
                    or ""
                )
                # NB: assets values are AssetState (Pydantic), not dicts → normalize.
                primary_asset_update = (
                    _as_dict((state_update.get("assets") or {}).get(primary_sym))
                    if primary_sym else {}
                )

                # Add node-specific data (lê do update direto OU do AssetState do primary)
                if node_name == "orchestrator":
                    if "next_agent" in state_update:
                        step_data["next_agent"] = state_update["next_agent"]
                    if primary_sym:
                        step_data["symbol"] = primary_sym
                    if state_update.get("symbols"):
                        step_data["symbols"] = state_update["symbols"]
                    if "timeframe" in state_update:
                        step_data["timeframe"] = str(state_update["timeframe"])

                elif node_name == "market_data":
                    if primary_asset_update.get("live_price") is not None:
                        step_data["live_price"] = primary_asset_update["live_price"]
                    if primary_asset_update.get("volume_24h") is not None:
                        step_data["volume_24h"] = primary_asset_update["volume_24h"]

                elif node_name in ("features_macro", "features_setup", "features_exec", "feature_engineering"):
                    if primary_asset_update.get("macro_rsi_14") is not None:
                        step_data["rsi"] = primary_asset_update["macro_rsi_14"]
                    elif primary_asset_update.get("setup_rsi_14") is not None:
                        step_data["rsi"] = primary_asset_update["setup_rsi_14"]
                    elif primary_asset_update.get("exec_rsi_14") is not None:
                        step_data["rsi"] = primary_asset_update["exec_rsi_14"]
                    if primary_asset_update.get("macro_macd_line") is not None:
                        step_data["macd"] = primary_asset_update["macro_macd_line"]

                elif node_name == "risk_agent":
                    if primary_asset_update.get("risk_level") is not None:
                        step_data["risk_level"] = str(primary_asset_update["risk_level"])
                    if primary_asset_update.get("cvar_95") is not None:
                        step_data["cvar_95"] = primary_asset_update["cvar_95"]
                    if primary_asset_update.get("sharpe") is not None:
                        step_data["sharpe"] = primary_asset_update["sharpe"]

                elif node_name == "signal_agent":
                    if primary_asset_update.get("signal_direction") is not None:
                        step_data["signal_direction"] = primary_asset_update["signal_direction"]
                    if primary_asset_update.get("signal_confidence") is not None:
                        step_data["signal_confidence"] = primary_asset_update["signal_confidence"]

                elif node_name == "risk_gate":
                    if "gate_approved" in state_update:
                        step_data["gate_approved"] = state_update["gate_approved"]
                    if "gate_reason" in state_update:
                        step_data["gate_reason"] = state_update["gate_reason"]
                    if "next_action" in state_update:
                        step_data["next_action"] = str(state_update["next_action"])
                
                # Emit SSE event
                yield f"data: {json.dumps(step_data)}\n\n"
        
        # ── Multi-asset view for the final event ──
        primary_sym = accumulated_state.get("primary_symbol") or ""
        all_syms = accumulated_state.get("symbols", []) or ([primary_sym] if primary_sym else [])
        assets_dict = accumulated_state.get("assets", {}) or {}
        # AssetState (Pydantic) → dict for uniform .get() access.
        primary = _as_dict(assets_dict.get(primary_sym))

        compared: list[dict] = []
        for s in all_syms:
            if not s or s == primary_sym:
                continue
            asset = _as_dict(assets_dict.get(s))
            compared.append({
                "symbol": s,
                "live_price": asset.get("live_price"),
                "risk_level": str(asset["risk_level"]) if asset.get("risk_level") else None,
                "sharpe": asset.get("sharpe"),
                "max_drawdown": asset.get("max_drawdown"),
                "forecast_return_pct": asset.get("forecast_return_pct"),
            })

        final_event = {
            "node": "final",
            "type": "completion",
            "response": accumulated_state.get("final_answer", ""),
            "pipeline": {
                "symbol": primary_sym,
                "symbols": all_syms,
                "timeframe": str(accumulated_state.get("timeframe")),
                "risk_level": str(primary["risk_level"]) if primary.get("risk_level") else None,
                "signal_direction": primary.get("signal_direction"),
                "signal_confidence": primary.get("signal_confidence"),
                "gate_approved": accumulated_state.get("gate_approved"),
                "gate_reason": accumulated_state.get("gate_reason"),
                "anomalies_detected": primary.get("anomalies_detected", []),
                "compared": compared,
            },
            "timestamp": asyncio.get_event_loop().time(),
            "session_id": session_id,
        }
        
        # Save assistant response with complete metadata
        await conv_history.save_message(
            user_id=request.user_id,
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=accumulated_state.get("final_answer", ""),
            metadata={
                "pipeline_summary": final_event["pipeline"],
                "chain_of_thought": accumulated_state.get("cot", ""),
            },
        )
        
        yield f"data: {json.dumps(final_event)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
