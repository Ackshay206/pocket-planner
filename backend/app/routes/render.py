"""
Render Route

POST /render - Generate an edited image with the optimized layout.
Uses Gemini image editing to visualize furniture movements.
"""

from fastapi import APIRouter, HTTPException
import asyncio

from app.models.api import RenderRequest, RenderResponse
from app.models.room import RoomObject
from app.agents.render_node import ImageEditor, EditMask


router = APIRouter(prefix="/render", tags=["Rendering"])


@router.post("", response_model=RenderResponse)
async def render_layout(request: RenderRequest) -> RenderResponse:
    """
    Generate an edited image showing the optimized layout.
    
    This endpoint:
    1. Compares original and final layouts
    2. Generates image edit prompts for moved objects
    3. Uses Gemini to edit the image
    4. Returns the rendered result
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
    
    # Generate edit instructions from layout changes
    edit_masks = []
    for change in changes:
        # Create an instruction describing the furniture movement
        instruction = (
            f"Move the {change['label']} from its current position "
            f"to the new location. Keep the same furniture style and lighting."
        )
        
        # For now, we'll use a simple region description
        # In a full implementation, we'd generate actual mask images
        edit_masks.append(EditMask(
            region_mask=request.original_image_base64[:100] + "==",  # Placeholder mask
            instruction=instruction
        ))
    
    # Try to apply edits using Gemini
    try:
        editor = ImageEditor()
        
        # Apply surgical edits
        edited_image_base64 = await editor.apply_edits(
            base_image=request.original_image_base64,
            masks=edit_masks
        )
        
        change_descriptions = [
            f"Moved {c['label']} from ({c['from'][0]}, {c['from'][1]}) to ({c['to'][0]}, {c['to'][1]})"
            for c in changes
        ]
        
        return RenderResponse(
            image_url=None,
            image_base64=edited_image_base64,
            message=f"Applied {len(changes)} change(s): " + "; ".join(change_descriptions)
        )
        
    except Exception as e:
        # If Gemini editing fails, return a descriptive message
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
async def get_render_status(job_id: str):
    """
    Check status of an async render job.
    
    For long-running renders, this allows polling for completion.
    """
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Async rendering not yet implemented"
    }
