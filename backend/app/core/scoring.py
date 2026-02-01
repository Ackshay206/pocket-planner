"""
Layout Scoring System

Calculates a 0-100 score for room layouts based on:
- Constraint satisfaction (hard constraints)
- Walkability (path accessibility)
- Preferences (soft constraints)
- Space efficiency
"""

from typing import List, Tuple
from dataclasses import dataclass

from app.models.room import RoomObject, LayoutScore
from app.core.geometry import (
    calculate_furniture_density,
    get_free_space,
    find_collisions
)
from app.core.constraints import (
    check_all_hard_constraints,
    evaluate_soft_constraints
)


# Score weights (must sum to 1.0)
WEIGHT_CONSTRAINTS = 0.40    # Hard constraint satisfaction
WEIGHT_WALKABILITY = 0.30    # Path accessibility
WEIGHT_PREFERENCES = 0.20    # Soft constraints
WEIGHT_EFFICIENCY = 0.10     # Space usage efficiency


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of layout score components."""
    constraint_score: float      # 0-100
    walkability_score: float     # 0-100
    preference_score: float      # 0-100
    efficiency_score: float      # 0-100
    total_score: float           # 0-100 weighted
    violations_count: int
    suggestions: List[str]


def calculate_constraint_score(
    objects: List[RoomObject],
    room_width: int,
    room_height: int
) -> Tuple[float, int]:
    """
    Calculate score based on hard constraint violations.
    
    Returns:
        (score, violation_count)
    """
    violations = check_all_hard_constraints(objects, room_width, room_height)
    
    if len(violations) == 0:
        return (100.0, 0)
    
    # Each violation reduces score
    # Max penalty at 5+ violations
    penalty_per_violation = 20.0
    penalty = min(len(violations) * penalty_per_violation, 100.0)
    
    return (100.0 - penalty, len(violations))


def calculate_walkability_score(
    objects: List[RoomObject],
    room_width: int,
    room_height: int
) -> float:
    """
    Calculate score based on available walking space.
    
    A good layout has:
    - At least 30% of room as free space
    - Clear paths between door and key furniture
    """
    # Get free space ratio
    free_space = get_free_space(room_width, room_height, objects)
    room_area = room_width * room_height
    
    if room_area == 0:
        return 0.0
    
    free_ratio = free_space.area / room_area
    
    # Ideal: 30-50% free space
    if free_ratio >= 0.30:
        space_score = 100.0
    elif free_ratio >= 0.20:
        space_score = 70.0
    elif free_ratio >= 0.10:
        space_score = 40.0
    else:
        space_score = 20.0
    
    # Check for collisions (reduces walkability)
    collisions = find_collisions(objects)
    if collisions:
        space_score -= len(collisions) * 15.0
    
    return max(0.0, space_score)


def calculate_efficiency_score(
    objects: List[RoomObject],
    room_width: int,
    room_height: int
) -> float:
    """
    Calculate score based on space usage efficiency.
    
    Penalizes:
    - Too much empty space (underutilized)
    - Too cluttered (overutilized)
    """
    density = calculate_furniture_density(room_width, room_height, objects)
    
    # Ideal density: 40-60%
    if 40 <= density <= 60:
        return 100.0
    elif 30 <= density < 40 or 60 < density <= 70:
        return 80.0
    elif 20 <= density < 30 or 70 < density <= 80:
        return 60.0
    else:
        return 40.0


def score_layout(
    objects: List[RoomObject],
    room_width: int,
    room_height: int
) -> LayoutScore:
    """
    Calculate overall layout score with breakdown.
    
    Args:
        objects: All room objects
        room_width: Room width in units
        room_height: Room height in units
        
    Returns:
        LayoutScore with total and component scores
    """
    # Calculate individual scores
    constraint_score, violations = calculate_constraint_score(
        objects, room_width, room_height
    )
    
    walkability_score = calculate_walkability_score(
        objects, room_width, room_height
    )
    
    preference_score, suggestions = evaluate_soft_constraints(objects)
    
    efficiency_score = calculate_efficiency_score(
        objects, room_width, room_height
    )
    
    # Calculate weighted total
    total = (
        constraint_score * WEIGHT_CONSTRAINTS +
        walkability_score * WEIGHT_WALKABILITY +
        preference_score * WEIGHT_PREFERENCES +
        efficiency_score * WEIGHT_EFFICIENCY
    )
    
    # Build explanation
    explanation_parts = []
    
    if violations > 0:
        explanation_parts.append(f"{violations} constraint violation(s) detected")
    
    if walkability_score < 50:
        explanation_parts.append("Walking space is limited")
    
    if suggestions:
        explanation_parts.extend(suggestions)
    
    if not explanation_parts:
        explanation_parts.append("Layout is well-optimized")
    
    return LayoutScore(
        total_score=round(total, 1),
        constraint_score=round(constraint_score, 1),
        walkability_score=round(walkability_score, 1),
        preference_score=round(preference_score, 1),
        explanation=". ".join(explanation_parts)
    )


def compare_layouts(
    layout_a: List[RoomObject],
    layout_b: List[RoomObject],
    room_width: int,
    room_height: int
) -> Tuple[str, float]:
    """
    Compare two layouts and return which is better.
    
    Returns:
        ("A" or "B", score difference)
    """
    score_a = score_layout(layout_a, room_width, room_height)
    score_b = score_layout(layout_b, room_width, room_height)
    
    diff = score_a.total_score - score_b.total_score
    
    if diff > 0:
        return ("A", diff)
    elif diff < 0:
        return ("B", abs(diff))
    else:
        return ("TIE", 0.0)
