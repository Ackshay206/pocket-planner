"""
Constraint Engine

Defines and checks hard/soft constraints for room layouts.
- Hard constraints MUST be satisfied (e.g., door clearance)
- Soft constraints are preferences (e.g., desk near window)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from app.models.room import RoomObject, ObjectType, ConstraintViolation
from app.core.geometry import (
    check_overlap,
    calculate_clearance,
    is_path_blocked,
    get_buffered_polygon,
    object_to_polygon
)


# ============ Constraint Definitions ============

class ConstraintSeverity(str, Enum):
    ERROR = "error"      # Must fix
    WARNING = "warning"  # Should fix


@dataclass
class HardConstraint:
    """A constraint that MUST be satisfied."""
    name: str
    description: str
    min_clearance: float  # Minimum distance in units


@dataclass
class SoftConstraint:
    """A constraint that is preferred but not required."""
    name: str
    description: str
    weight: float  # Impact on score (0.0 - 1.0)


# Default constraint values
DOOR_CLEARANCE = 60.0      # 60 units clearance for door swing
WALKING_PATH_WIDTH = 45.0  # 45 units minimum walking path
WINDOW_PROXIMITY = 100.0   # Desk should be within 100 units of window


HARD_CONSTRAINTS = [
    HardConstraint(
        name="door_clearance",
        description="Door must remain unblocked with swing path clear",
        min_clearance=DOOR_CLEARANCE
    ),
    HardConstraint(
        name="walking_path",
        description="Minimum walking clearance between objects",
        min_clearance=WALKING_PATH_WIDTH
    ),
    HardConstraint(
        name="no_overlap",
        description="Objects cannot overlap each other",
        min_clearance=0
    ),
    HardConstraint(
        name="locked_objects",
        description="Locked objects cannot be moved",
        min_clearance=0
    )
]

SOFT_CONSTRAINTS = [
    SoftConstraint(
        name="desk_near_window",
        description="Desk should be positioned near natural light",
        weight=0.3
    ),
    SoftConstraint(
        name="bed_away_from_door",
        description="Bed should be away from main entry",
        weight=0.2
    ),
    SoftConstraint(
        name="nightstand_near_bed",
        description="Nightstand should be adjacent to bed",
        weight=0.2
    )
]


# ============ Hard Constraint Checkers ============

def check_door_clearance(
    objects: List[RoomObject],
    min_clearance: float = DOOR_CLEARANCE
) -> List[ConstraintViolation]:
    """
    Check that all doors have sufficient clearance for opening.
    
    Returns list of violations if any furniture blocks door swing area.
    """
    violations = []
    
    doors = [obj for obj in objects if obj.label == "door"]
    movable_objects = [obj for obj in objects if obj.type == ObjectType.MOVABLE]
    
    for door in doors:
        # Create buffer zone around door for swing clearance
        door_zone = get_buffered_polygon(door, min_clearance)
        
        for obj in movable_objects:
            obj_poly = object_to_polygon(obj)
            if door_zone.intersects(obj_poly):
                violations.append(ConstraintViolation(
                    constraint_name="door_clearance",
                    description=f"{obj.label} ({obj.id}) is blocking {door.id}. "
                               f"Minimum clearance: {min_clearance} units",
                    severity="error",
                    objects_involved=[door.id, obj.id]
                ))
    
    return violations


def check_no_overlap(objects: List[RoomObject]) -> List[ConstraintViolation]:
    """
    Check that no movable objects overlap each other.
    """
    violations = []
    
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1:]:
            if check_overlap(obj_a, obj_b):
                violations.append(ConstraintViolation(
                    constraint_name="no_overlap",
                    description=f"{obj_a.label} ({obj_a.id}) overlaps with "
                               f"{obj_b.label} ({obj_b.id})",
                    severity="error",
                    objects_involved=[obj_a.id, obj_b.id]
                ))
    
    return violations


def check_walking_paths(
    objects: List[RoomObject],
    room_width: int,
    room_height: int,
    min_path_width: float = WALKING_PATH_WIDTH
) -> List[ConstraintViolation]:
    """
    Check that there's a walkable path from door to key furniture.
    """
    violations = []
    
    doors = [obj for obj in objects if obj.label == "door"]
    beds = [obj for obj in objects if obj.label == "bed"]
    movable_obstacles = [obj for obj in objects if obj.type == ObjectType.MOVABLE]
    
    for door in doors:
        for bed in beds:
            blocked, blocker_id = is_path_blocked(
                door.center, 
                bed.center, 
                movable_obstacles,
                path_width=min_path_width
            )
            if blocked:
                blocker = next((o for o in objects if o.id == blocker_id), None)
                blocker_label = blocker.label if blocker else "unknown"
                violations.append(ConstraintViolation(
                    constraint_name="walking_path",
                    description=f"Path from {door.id} to {bed.id} is blocked by "
                               f"{blocker_label} ({blocker_id})",
                    severity="error",
                    objects_involved=[door.id, bed.id, blocker_id]
                ))
    
    return violations


def check_all_hard_constraints(
    objects: List[RoomObject],
    room_width: int,
    room_height: int
) -> List[ConstraintViolation]:
    """
    Run all hard constraint checks and return combined violations.
    """
    violations = []
    
    violations.extend(check_door_clearance(objects))
    violations.extend(check_no_overlap(objects))
    violations.extend(check_walking_paths(objects, room_width, room_height))
    
    return violations


# ============ Soft Constraint Checkers ============

def check_desk_near_window(
    objects: List[RoomObject],
    max_distance: float = WINDOW_PROXIMITY
) -> Tuple[bool, float]:
    """
    Check if desk is positioned near a window.
    
    Returns:
        (is_satisfied, score) where score is 0.0-1.0
    """
    desks = [obj for obj in objects if obj.label == "desk"]
    windows = [obj for obj in objects if obj.label == "window"]
    
    if not desks or not windows:
        return (True, 1.0)  # N/A, consider satisfied
    
    # Find minimum distance from any desk to any window
    min_distance = float('inf')
    for desk in desks:
        for window in windows:
            dist = calculate_clearance(desk, window)
            min_distance = min(min_distance, dist)
    
    if min_distance <= max_distance:
        # Score based on proximity (closer = better)
        score = 1.0 - (min_distance / max_distance)
        return (True, max(0.5, score))
    else:
        return (False, 0.3)


def check_bed_away_from_door(
    objects: List[RoomObject],
    min_distance: float = 50.0
) -> Tuple[bool, float]:
    """
    Check if bed is positioned away from the door.
    
    Returns:
        (is_satisfied, score) where score is 0.0-1.0
    """
    beds = [obj for obj in objects if obj.label == "bed"]
    doors = [obj for obj in objects if obj.label == "door"]
    
    if not beds or not doors:
        return (True, 1.0)
    
    # Check distance from each bed to nearest door
    for bed in beds:
        for door in doors:
            dist = calculate_clearance(bed, door)
            if dist < min_distance:
                return (False, 0.4)
    
    return (True, 1.0)


def evaluate_soft_constraints(
    objects: List[RoomObject]
) -> Tuple[float, List[str]]:
    """
    Evaluate all soft constraints and return weighted score.
    
    Returns:
        (score, list of suggestions)
    """
    total_weight = 0.0
    weighted_score = 0.0
    suggestions = []
    
    # Desk near window
    satisfied, score = check_desk_near_window(objects)
    weight = 0.3
    weighted_score += score * weight
    total_weight += weight
    if not satisfied:
        suggestions.append("Consider moving desk closer to the window for better lighting")
    
    # Bed away from door
    satisfied, score = check_bed_away_from_door(objects)
    weight = 0.2
    weighted_score += score * weight
    total_weight += weight
    if not satisfied:
        suggestions.append("Consider moving bed further from the door for privacy")
    
    # Normalize score to 0-100
    final_score = (weighted_score / total_weight) * 100 if total_weight > 0 else 100.0
    
    return (final_score, suggestions)
