"""
Room and Furniture Data Models

These Pydantic models define the core data structures for representing
rooms, furniture objects, and their properties. They serve as the
"contract" between all system components.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    """Classification of room objects."""
    MOVABLE = "movable"      # Can be repositioned (bed, desk, chair)
    STRUCTURAL = "structural" # Fixed in place (door, window, wall)


class RoomDimensions(BaseModel):
    """Estimated dimensions of the room in pixels/units."""
    width_estimate: int = Field(..., gt=0, description="Room width")
    height_estimate: int = Field(..., gt=0, description="Room height")


class RoomObject(BaseModel):
    """
    A single object in the room (furniture or structural element).
    
    Attributes:
        id: Unique identifier (e.g., "bed_1", "door_1")
        label: Human-readable name (e.g., "bed", "desk")
        bbox: Bounding box as [x, y, width, height]
        type: Whether the object is movable or structural
        orientation: Rotation in degrees (0, 90, 180, 270)
        is_locked: Whether the user has locked this object in place
    """
    id: str = Field(..., description="Unique object ID")
    label: str = Field(..., description="Object type label")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x, y, width, height]")
    type: ObjectType = Field(default=ObjectType.MOVABLE)
    orientation: int = Field(default=0, ge=0, lt=360)
    is_locked: bool = Field(default=False, description="User-locked status")
    
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
    
    This is the exact schema that Gemini must return when analyzing
    a room image. It contains room dimensions and all detected objects.
    """
    room_dimensions: RoomDimensions
    objects: List[RoomObject]


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
