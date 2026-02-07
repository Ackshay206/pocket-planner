"""
Render Route

POST /render - Generate an edited image with the optimized layout.
POST /render/perspective - Generate a photorealistic perspective view.

FULLY TRACED with LangSmith.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from app.models.room import RoomObject, RoomDimensions
from app.models.api import RenderRequest, RenderResponse
from app.tools.edit_image import EditImageTool
from app.agents.perspective_node import PerspectiveGenerator

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


router = APIRouter(prefix="/render", tags=["Rendering"])


# === Request/Response Models ===

class RenderRequest(BaseModel):
    """Request body for render endpoint."""
    original_image_base64: str
    final_layout: List[RoomObject]
    original_layout: List[RoomObject]


class RenderResponse(BaseModel):
    """Response from render endpoint."""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    message: str


class PerspectiveRequest(BaseModel):
    """Request body for perspective generation."""
    layout: List[RoomObject]
    room_dimensions: RoomDimensions
    style: str = "modern"
    view_angle: str = "corner"
    image_base64: Optional[str] = None  # Context image (layout thumbnail)
    layout_plan: Optional[dict] = None  # Semantic placement plan from designer


class PerspectiveResponse(BaseModel):
    """Response from perspective generation."""
    image_base64: Optional[str] = None
    message: str


# === Endpoints ===

@router.post("/perspective", response_model=PerspectiveResponse)
@traceable(name="generate_perspective_endpoint", run_type="chain", tags=["api", "render", "perspective"])
async def generate_perspective(request: PerspectiveRequest) -> PerspectiveResponse:
    """
    Generate a photorealistic perspective view of the room layout.
    
    Uses Gemini image generation to create an eye-level view
    of the room based on the furniture layout.
    
    TRACED: Full trace with Gemini image generation details.
    """
    try:
        generator = PerspectiveGenerator()
        
        image_base64 = await generator.generate_side_view(
            layout=request.layout,
            room_dims=request.room_dimensions,
            style=request.style,
            view_angle=request.view_angle,
            lighting="natural daylight",
            image_base64=request.image_base64,
            layout_plan=request.layout_plan,
        )
        
        return PerspectiveResponse(
            image_base64=image_base64,
            message="Perspective view generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Perspective generation failed: {str(e)}"
        )


@router.post("", response_model=RenderResponse)
@traceable(name="render_layout_endpoint", run_type="chain", tags=["api", "render", "edit"])
async def render_layout(request: RenderRequest) -> RenderResponse:
    """
    Generate an edited image showing the optimized layout.
    
    This endpoint:
    1. Compares original and final layouts
    2. Generates image edit prompts for moved objects
    3. Uses Gemini to edit the image
    4. Returns the rendered result
    
    TRACED: Full trace with image edit details.
    """
    # Calculate what changed
    changes = []
    original_positions = {obj.id: obj.bbox for obj in request.original_layout}
    
    for obj in request.final_layout:
        original_bbox = original_positions.get(obj.id)
        if original_bbox and original_bbox != obj.bbox:
            changes.append({
                "object_id": obj.id,
                "label": obj.label,
                "from": original_bbox,
                "to": obj.bbox
            })
    
    if not changes:
        return RenderResponse(
            image_url=None,
            image_base64=None,
            message="No changes to render. Layout is unchanged."
        )
    
    # Try to apply edits using Gemini
    try:
        editor = EditImageTool()
        current_image = request.original_image_base64
        
        change_descriptions = []
        
        for change in changes:
            instruction = (
                f"Move the {change['label']} from its current position "
                f"to the new location. Keep the same furniture style and lighting."
            )
            
            current_image = await editor.edit_image(
                base_image=current_image,
                instruction=instruction
            )
            
            change_descriptions.append(
                f"Moved {change['label']} from ({change['from'][0]}, {change['from'][1]}) "
                f"to ({change['to'][0]}, {change['to'][1]})"
            )
        
        return RenderResponse(
            image_url=None,
            image_base64=current_image,
            message=f"Applied {len(changes)} change(s): " + "; ".join(change_descriptions)
        )
        
    except Exception as e:
        change_descriptions = [
            f"Move {c['label']} from ({c['from'][0]}, {c['from'][1]}) to ({c['to'][0]}, {c['to'][1]})"
            for c in changes
        ]
        
        return RenderResponse(
            image_url=None,
            image_base64=None,
            message=f"Render requested for {len(changes)} change(s): " + "; ".join(change_descriptions) + f" (Note: {str(e)})"
        )


@router.get("/status/{job_id}")
@traceable(name="get_render_status", run_type="retriever", tags=["api", "render", "status"])
async def get_render_status(job_id: str):
    """Check status of an async render job."""
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Async rendering not yet implemented"
    }