"""
Chat Route

POST /chat/edit - Process conversational edit commands for room layouts and images.

FULLY TRACED with LangSmith.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.models.room import RoomObject, RoomDimensions
from app.agents.chat_editor_node import ChatEditor

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_ENABLED = True
except ImportError:
    LANGSMITH_ENABLED = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatEditRequest(BaseModel):
    """Request body for chat edit endpoint."""
    command: str
    current_layout: List[RoomObject]
    room_dimensions: RoomDimensions
    current_image_base64: Optional[str] = None
    layout_plan: Optional[dict] = None


class ChatEditResponse(BaseModel):
    """Response from chat edit endpoint."""
    edit_type: str  # 'layout', 'cosmetic', 'replace', 'remove', or 'error'
    updated_layout: List[RoomObject]
    updated_image_base64: Optional[str] = None
    explanation: str
    needs_rerender: bool = False


@router.post("/edit", response_model=ChatEditResponse)
@traceable(
    name="chat_edit_endpoint", 
    run_type="chain", 
    tags=["api", "chat", "edit"],
    metadata={"description": "Process natural language editing commands"}
)
async def chat_edit(request: ChatEditRequest) -> ChatEditResponse:
    """
    Process a natural language editing command.
    
    Commands can be:
    - Layout edits: "move desk to the left", "rotate bed 90 degrees"
    - Cosmetic edits: "make it more cozy", "add plants"
    - Remove edits: "remove the desk", "get rid of the nightstand"
    
    Returns updated layout and/or image based on edit type.
    
    TRACED: Full trace with command parsing and edit application.
    """
    try:
        editor = ChatEditor()
        
        result = await editor.process_edit_command(
            command=request.command,
            current_layout=request.current_layout,
            room_dims=request.room_dimensions,
            current_image_base64=request.current_image_base64,
            layout_plan=request.layout_plan
        )
        
        return ChatEditResponse(
            edit_type=result.get("edit_type", "error"),
            updated_layout=result.get("updated_layout", request.current_layout),
            updated_image_base64=result.get("updated_image_base64"),
            explanation=result.get("explanation", "Edit processed"),
            needs_rerender=result.get("needs_rerender", False)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat edit failed: {str(e)}"
        )