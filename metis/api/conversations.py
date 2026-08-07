"""API endpoints for conversation history management."""

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from metis.memory.conversation_history import (
    get_conversation_history,
    FeedbackRating,
)
from metis.jwt_verifier import JWTVerifier, InvalidTokenError
from metis.api.deps import get_jwt_verifier
import uuid

router = APIRouter()


# Request/Response Models
class FeedbackRequest(BaseModel):
    rating: FeedbackRating
    comment: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    title: str = ""
    message_count: int
    first_message: str
    last_message: str
    has_feedback: bool


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict
    feedback: Optional[str] = None
    feedback_rating: Optional[str] = None
    feedback_comment: Optional[str] = None


class ConversationResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]


async def _require_matching_user(
    user_id: str,
    authorization: Optional[str] = Header(None),
    verifier: JWTVerifier = Depends(get_jwt_verifier),
) -> str:
    """Validate the JWT and ensure the token's user_id matches the path user_id.

    Prevents IDOR — a user cannot access another user's conversations by
    changing the path parameter.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    try:
        identity = await verifier.verify(token)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if identity.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: token does not match requested user")

    return user_id


@router.get("/conversations/{user_id}", response_model=List[SessionResponse])
async def list_conversations(
    user_id: str,
    _: str = Depends(_require_matching_user),
    limit: int = 20,
):
    """List all conversation sessions for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of sessions to return
        
    Returns:
        List of conversation sessions
    """
    conv_history = get_conversation_history()
    sessions = await conv_history.get_conversation_sessions(
        user_id=user_id,
        limit=limit,
    )
    return sessions


@router.get("/conversations/{user_id}/{session_id}", response_model=ConversationResponse)
async def get_conversation(
    user_id: str,
    session_id: str,
    _: str = Depends(_require_matching_user),
    limit: int = 100,
):
    """Get all messages in a specific conversation session.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        limit: Maximum number of messages to return
        
    Returns:
        Conversation with all messages
    """
    conv_history = get_conversation_history()
    messages = await conv_history.get_conversation_history(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
    )
    
    return ConversationResponse(
        session_id=session_id,
        messages=[MessageResponse(**msg) for msg in messages],
    )


@router.delete("/conversations/{user_id}/{session_id}")
async def delete_conversation(
    user_id: str,
    session_id: str,
    _: str = Depends(_require_matching_user),
):
    """Soft delete a conversation session.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        Number of messages deleted
    """
    conv_history = get_conversation_history()
    deleted_count = await conv_history.soft_delete_conversation(
        user_id=user_id,
        session_id=session_id,
    )
    
    return {"deleted_count": deleted_count}


@router.post("/conversations/{user_id}/{session_id}/feedback")
async def feedback_conversation(
    user_id: str,
    session_id: str,
    feedback: FeedbackRequest,
    _: str = Depends(_require_matching_user),
):
    """Add feedback to an entire conversation session.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        feedback: Feedback data (rating, comment)
        
    Returns:
        Number of messages updated with feedback
    """
    conv_history = get_conversation_history()
    updated_count = await conv_history.feedback_conversation(
        user_id=user_id,
        session_id=session_id,
        rating=feedback.rating,
        comment=feedback.comment,
    )
    
    return {"updated_count": updated_count}


@router.post("/conversations/{user_id}/{session_id}/messages/{message_id}/feedback")
async def feedback_message(
    user_id: str,
    session_id: str,
    message_id: str,
    feedback: FeedbackRequest,
    _: str = Depends(_require_matching_user),
):
    """Add feedback to a specific message.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        message_id: Message identifier
        feedback: Feedback data (rating, comment)
        
    Returns:
        Success status
    """
    conv_history = get_conversation_history()
    success = await conv_history.feedback_message(
        user_id=user_id,
        message_id=message_id,
        rating=feedback.rating,
        comment=feedback.comment,
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Message not found or unauthorized")
    
    return {"success": True}
