from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.agent.graph import get_agent_graph

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent_graph()
    state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "next_action": "model",
        "intermediate_steps": [],
        "final_answer": None
    }
    final_state = await agent.ainvoke(state, config={"recursion_limit": 20})
    return {
        "response": final_state["final_answer"],
        "tools_used": [step[0] for step in final_state["intermediate_steps"]]
    }
