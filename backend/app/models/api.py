"""
API Request/Response Schemas

Pydantic models for API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.room import RoomObject, RoomDimensions, ConstraintViolation, LayoutScore


# ============ Analyze Endpoint ============

class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint (base64 image)."""
    image_base64: str = Field(..., description="Room photo as base64 string")


class AnalyzeResponse(BaseModel):
    """Response from /analyze endpoint."""
    room_dimensions: RoomDimensions
    objects: List[RoomObject]
    detected_issues: List[str] = Field(default_factory=list)
    message: str = "Analysis complete"


# ============ Optimize Endpoint ============

class OptimizeRequest(BaseModel):
    """Request body for /optimize endpoint."""
    current_layout: List[RoomObject] = Field(..., description="Current furniture positions")
    locked_ids: List[str] = Field(default_factory=list, description="IDs of locked objects")
    room_dimensions: RoomDimensions = Field(..., description="Room size")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Max optimization iterations")


class OptimizeResponse(BaseModel):
    """Response from /optimize endpoint."""
    new_layout: List[RoomObject]
    explanation: str
    layout_score: float
    iterations: int
    constraint_violations: List[ConstraintViolation] = Field(default_factory=list)
    improvement: float = Field(default=0.0, description="Score improvement from original")


# ============ Render Endpoint ============

class RenderRequest(BaseModel):
    """Request body for /render endpoint."""
    original_image_base64: str = Field(..., description="Original room photo")
    final_layout: List[RoomObject] = Field(..., description="Target furniture positions")
    original_layout: List[RoomObject] = Field(..., description="Original positions for diff")


class RenderResponse(BaseModel):
    """Response from /render endpoint."""
    image_url: Optional[str] = Field(None, description="URL to rendered image")
    image_base64: Optional[str] = Field(None, description="Rendered image as base64")
    message: str = "Render complete"


# ============ Health Check ============

class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str = "ok"
    version: str
    message: str = "Pocket Planner API is running"


# ============ Error Response ============

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
    context: Optional[dict] = None
