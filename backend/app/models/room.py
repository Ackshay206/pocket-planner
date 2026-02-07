"""
Room and Furniture Data Models

These Pydantic models define the core data structures for representing
rooms, furniture objects, and their properties. They serve as the
"contract" between all system components.
"""

from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    """Classification of room objects."""
    MOVABLE = "movable"      # Can be repositioned (bed, desk, chair)
    STRUCTURAL = "structural" # Fixed in place (door, window, wall)


class RoomDimensions(BaseModel):
    """Estimated dimensions of the room in pixels/units."""
    width_estimate: float = Field(..., gt=0, description="Room width")
    height_estimate: float = Field(..., gt=0, description="Room height")


class RoomObject(BaseModel):
    """
    A single object in the room (furniture or structural element).
    
    Attributes:
        id: Unique identifier (e.g., "bed_1", "door_1")
        label: Human-readable name (e.g., "bed", "desk")
        bbox: Bounding box as [x, y, width, height]
        type: Whether the object is movable or structural
        orientation: Rotation in degrees (0=North, 90=East, 180=South, 270=West)
        is_locked: Whether the user has locked this object in place
        z_index: Depth ordering (0=floor/rugs, 1=furniture, 2=ceiling)
        material_hint: Material type for rendering (wooden, fabric, metal, glass)
        footprint_polygon: Optional precise polygon for L-shaped/curved objects
    """
    id: str = Field(..., description="Unique object ID")
    label: str = Field(..., description="Object type label")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x, y, width, height]")
    type: ObjectType = Field(default=ObjectType.MOVABLE)
    orientation: int = Field(default=0, ge=0, lt=360, description="0=North, 90=East, 180=South, 270=West")
    is_locked: bool = Field(default=False, description="User-locked status")
    
    # NEW FIELDS for 3D floor plan understanding
    z_index: int = Field(default=1, ge=0, le=2, description="0=floor/rugs, 1=furniture, 2=ceiling")
    material_hint: Optional[str] = Field(None, description="e.g., 'wooden', 'fabric', 'glass', 'metal'")
    footprint_polygon: Optional[List[Tuple[float, float]]] = Field(
        None, 
        description="Optional precise polygon for L-shaped/curved objects as [(x1,y1), (x2,y2), ...]"
    )
    
    @property
    def x(self) -> int:
        """X coordinate of top-left corner."""
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        """Y coordinate of top-left corner."""
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        """Object width."""
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        """Object height."""
        return self.bbox[3]
    
    @property
    def center(self) -> tuple[int, int]:
        """Center point of the object."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class VisionOutput(BaseModel):
    """
    Output from the Vision Node (Gemini analysis of room photo).
    """
    room_dimensions: RoomDimensions
    objects: List[RoomObject]
    wall_bounds: Optional[List[int]] = Field(
        default=None,
        description="Interior wall boundaries as [x, y, width, height]"
    )
    # NEW: image pixel dimensions for accurate spatial calculations
    image_width: Optional[int] = Field(default=None, description="Source image width in pixels")
    image_height: Optional[int] = Field(default=None, description="Source image height in pixels")


class ConstraintViolation(BaseModel):
    """A single constraint violation detected in the layout."""
    constraint_name: str = Field(..., description="Name of violated constraint")
    description: str = Field(..., description="Human-readable explanation")
    severity: str = Field(default="error", description="'error' or 'warning'")
    objects_involved: List[str] = Field(default_factory=list, description="IDs of objects involved")


class LayoutScore(BaseModel):
    """Scoring result for a room layout."""
    total_score: float = Field(..., ge=0, le=100, description="Overall score 0-100")
    walkability_score: float = Field(..., ge=0, le=100)
    constraint_score: float = Field(..., ge=0, le=100)
    preference_score: float = Field(..., ge=0, le=100)
    explanation: str = Field(default="", description="Summary of scoring factors")
