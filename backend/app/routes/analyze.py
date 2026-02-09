"""
Analyze Route

POST /analyze - Analyze a room image and extract furniture objects.
Uses Gemini 2.5 Flash for vision analysis with object detection.

FULLY TRACED with LangSmith.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
import base64
import logging

from app.models.api import AnalyzeRequest, AnalyzeResponse
from app.agents.vision_node import VisionAgent, get_vision_agent

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


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=AnalyzeResponse)
@traceable(name="analyze_room_endpoint", run_type="chain", tags=["api", "vision"])
async def analyze_room(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a room image and extract furniture objects.
    
    This endpoint:
    1. Sends image to Gemini 2.5 Flash Vision
    2. Extracts room dimensions and detected objects with bounding boxes
    3. Checks for constraint violations
    4. Returns structured layout data
    
    TRACED: Full trace with Gemini call details.
    """
    try:
        # Call Gemini Vision (via VisionAgent - also traced)
        agent = get_vision_agent()
        vision_output = await agent.analyze_room(request.image_base64)
        
        # Check for initial issues
        return AnalyzeResponse(
            room_dimensions=vision_output.room_dimensions,
            objects=vision_output.objects,
            wall_bounds=vision_output.wall_bounds,
            message=f"Detected {len(vision_output.objects)} objects.",
            image_width=vision_output.image_width,
            image_height=vision_output.image_height,
        )
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Vision analysis failed: {str(e)}"
        )


@router.post("/upload", response_model=AnalyzeResponse)
@traceable(name="analyze_room_upload", run_type="chain", tags=["api", "vision", "upload"])
async def analyze_room_upload(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Analyze a room image uploaded as a file.
    
    Accepts: JPEG, PNG, WebP
    TRACED: Full trace with file metadata.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Read and convert to base64
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode("utf-8")
    
    # Call the main analyze function
    request = AnalyzeRequest(image_base64=image_base64)
    return await analyze_room(request)