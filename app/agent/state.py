from typing import List, Tuple, TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # acumula mensagens
    user_id: str
    next_action: str                # "tools" | "finalize"
    intermediate_steps: List[Tuple[str, str]]  # (tool_name, result)
    final_answer: Optional[str]

