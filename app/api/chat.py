from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import get_agent_graph, QuantAgentState
from app.memory.conversation_history import (
    get_conversation_history,
    MessageRole,
)
import json
import asyncio
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

@router.post("/chat")
async def chat(request: ChatRequest):
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
    
    agent = get_agent_graph()
    initial_state = QuantAgentState(
        messages=messages,
        user_id=request.user_id,
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
async def streaming_chat(request: ChatRequest):
    """Streaming endpoint that emits each node execution as an SSE event."""
    
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
        
        agent = get_agent_graph()
        initial_state = QuantAgentState(
            messages=messages,
            user_id=request.user_id,
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
                
                # Add node-specific data
                if node_name == "orchestrator":
                    if "next_agent" in state_update:
                        step_data["next_agent"] = state_update["next_agent"]
                    if "symbol" in state_update:
                        step_data["symbol"] = state_update["symbol"]
                    if "timeframe" in state_update:
                        step_data["timeframe"] = str(state_update["timeframe"])
                
                elif node_name == "market_data":
                    if "live_price" in state_update:
                        step_data["live_price"] = state_update["live_price"]
                    if "volume_24h" in state_update:
                        step_data["volume_24h"] = state_update["volume_24h"]
                
                elif node_name == "feature_engineering":
                    if "rsi" in state_update:
                        step_data["rsi"] = state_update["rsi"]
                    if "macd" in state_update:
                        step_data["macd"] = state_update["macd"]
                    if "bb_upper" in state_update:
                        step_data["bb_upper"] = state_update["bb_upper"]
                
                elif node_name == "risk_agent":
                    if "risk_level" in state_update:
                        step_data["risk_level"] = str(state_update["risk_level"])
                    if "cvar_95" in state_update:
                        step_data["cvar_95"] = state_update["cvar_95"]
                    if "sharpe" in state_update:
                        step_data["sharpe"] = state_update["sharpe"]
                
                elif node_name == "signal_agent":
                    if "signal_direction" in state_update:
                        step_data["signal_direction"] = state_update["signal_direction"]
                    if "signal_confidence" in state_update:
                        step_data["signal_confidence"] = state_update["signal_confidence"]
                
                elif node_name == "risk_gate":
                    if "gate_approved" in state_update:
                        step_data["gate_approved"] = state_update["gate_approved"]
                    if "gate_reason" in state_update:
                        step_data["gate_reason"] = state_update["gate_reason"]
                    if "next_action" in state_update:
                        step_data["next_action"] = str(state_update["next_action"])
                
                # Emit SSE event
                yield f"data: {json.dumps(step_data)}\n\n"
        
        # Send final event with complete accumulated state
        final_event = {
            "node": "final",
            "type": "completion",
            "response": accumulated_state.get("final_answer", ""),
            "pipeline": {
                "symbol": accumulated_state.get("symbol"),
                "timeframe": str(accumulated_state.get("timeframe")),
                "risk_level": str(accumulated_state.get("risk_level")),
                "signal_direction": accumulated_state.get("signal_direction"),
                "signal_confidence": accumulated_state.get("signal_confidence"),
                "gate_approved": accumulated_state.get("gate_approved"),
                "gate_reason": accumulated_state.get("gate_reason"),
                "anomalies_detected": accumulated_state.get("anomalies_detected", []),
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
