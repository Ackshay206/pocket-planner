"""
Analyze Route

POST /analyze - Analyze a room image and extract furniture objects.
Note: Vision logic is a placeholder for Developer A.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
import base64

from app.models.api import AnalyzeRequest, AnalyzeResponse
from app.models.room import RoomObject, RoomDimensions, ObjectType
from app.core.constraints import check_all_hard_constraints


router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=AnalyzeResponse)
async def analyze_room(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a room image and extract furniture objects.
    
    This endpoint:
    1. Sends image to Gemini Vision (placeholder)
    2. Extracts room dimensions and objects
    3. Returns structured layout data
    
    Note: Vision logic will be implemented by Developer A.
    Currently returns mock data for testing.
    """
    # === PLACEHOLDER: Developer A will implement Gemini Vision call ===
    # For now, return mock data to allow frontend development
    
    mock_objects = [
        RoomObject(
            id="door_1",
            label="door",
            bbox=[0, 150, 20, 80],
            type=ObjectType.STRUCTURAL,
            orientation=0
        ),
        RoomObject(
            id="window_1",
            label="window",
            bbox=[250, 0, 50, 20],
            type=ObjectType.STRUCTURAL,
            orientation=0
        ),
        RoomObject(
            id="bed_1",
            label="bed",
            bbox=[100, 200, 120, 180],
            type=ObjectType.MOVABLE,
            orientation=0
        ),
        RoomObject(
            id="desk_1",
            label="desk",
            bbox=[50, 50, 80, 50],
            type=ObjectType.MOVABLE,
            orientation=0
        ),
        RoomObject(
            id="chair_1",
            label="chair",
            bbox=[60, 110, 40, 40],
            type=ObjectType.MOVABLE,
            orientation=0
        ),
    ]
    
    mock_dimensions = RoomDimensions(
        width_estimate=300,
        height_estimate=400
    )
    
    # Check for initial issues
    violations = check_all_hard_constraints(
        mock_objects,
        mock_dimensions.width_estimate,
        mock_dimensions.height_estimate
    )
    
    detected_issues = [v.description for v in violations]
    
    return AnalyzeResponse(
        room_dimensions=mock_dimensions,
        objects=mock_objects,
        detected_issues=detected_issues,
        message=f"Detected {len(mock_objects)} objects. {len(detected_issues)} issue(s) found."
    )


@router.post("/upload", response_model=AnalyzeResponse)
async def analyze_room_upload(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Analyze a room image uploaded as a file.
    
    Accepts: JPEG, PNG, WebP
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
