"""
Solver Node

LangGraph node that attempts to fix constraint violations
by repositioning unlocked furniture.
"""

from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy

from app.models.state import AgentState
from app.models.room import RoomObject, ObjectType, ConstraintViolation
from app.core.geometry import (
    check_overlap,
    calculate_clearance,
    get_free_space,
    bbox_to_polygon,
    check_room_bounds
)
from app.core.room_graph import RoomGraph
from app.core.scoring import score_layout


# Movement step size
MOVE_STEP = 10  # Units per movement attempt


def solver_node(state: AgentState) -> Dict[str, Any]:
    """
    Attempt to fix constraint violations by moving unlocked objects.
    
    This node:
    1. Identifies which objects need to move
    2. Tries different positions for each
    3. Keeps changes that improve the score
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with new layout and explanation
    """
    current_layout = deepcopy(state["current_layout"])
    violations = state.get("constraint_violations", [])
    locked_ids = set(state.get("locked_object_ids", []))
    room_dims = state["room_dimensions"]
    
    if not violations:
        return {
            "iteration_count": state["iteration_count"] + 1,
            "explanation": "No violations to fix."
        }
    
    # Build room graph
    graph = RoomGraph()
    graph.add_objects(current_layout)
    for lid in locked_ids:
        graph.lock_object(lid)
    
    # Get objects to move (in dependency order)
    movable_ids = graph.get_movable_objects()
    
    # Track what we changed
    changes_made = []
    
    # Get initial score
    initial_score = score_layout(
        current_layout,
        room_dims.width_estimate,
        room_dims.height_estimate
    ).total_score
    
    # Try to fix each violation
    for violation in violations:
        objects_involved = violation.objects_involved
        
        # Find movable objects in this violation
        objects_to_move = [
            oid for oid in objects_involved 
            if oid in movable_ids
        ]
        
        for obj_id in objects_to_move:
            obj = next((o for o in current_layout if o.id == obj_id), None)
            if not obj:
                continue
            
            # Try moving in different directions
            new_pos = find_better_position(
                obj, 
                current_layout, 
                room_dims.width_estimate,
                room_dims.height_estimate,
                locked_ids
            )
            
            if new_pos:
                old_pos = obj.bbox.copy()
                obj.bbox[0] = new_pos[0]
                obj.bbox[1] = new_pos[1]
                changes_made.append(
                    f"Moved {obj.label} ({obj.id}) from ({old_pos[0]}, {old_pos[1]}) "
                    f"to ({new_pos[0]}, {new_pos[1]})"
                )
    
    # Calculate new score
    new_score = score_layout(
        current_layout,
        room_dims.width_estimate,
        room_dims.height_estimate
    ).total_score
    
    # Build explanation
    if changes_made:
        improvement = new_score - initial_score
        explanation = (
            f"Made {len(changes_made)} change(s). "
            f"Score: {initial_score:.1f} → {new_score:.1f} "
            f"({'+' if improvement >= 0 else ''}{improvement:.1f})\n\n"
            + "\n".join(changes_made)
        )
    else:
        explanation = "Could not find a better position for the furniture."
    
    return {
        "current_layout": current_layout,
        "iteration_count": state["iteration_count"] + 1,
        "explanation": explanation
    }


def find_better_position(
    obj: RoomObject,
    all_objects: List[RoomObject],
    room_width: int,
    room_height: int,
    locked_ids: set
) -> Optional[Tuple[int, int]]:
    """
    Find a better position for an object that reduces violations.
    
    Tries moving in cardinal directions with increasing distance.
    
    Returns:
        New (x, y) position or None if no improvement found
    """
    original_x, original_y = obj.x, obj.y
    best_position = None
    best_score = float('-inf')
    
    # Other objects to check against (excluding self)
    other_objects = [o for o in all_objects if o.id != obj.id]
    
    # Directions to try: right, down, left, up, and diagonals
    directions = [
        (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    # Try different distances
    for distance in [MOVE_STEP, MOVE_STEP * 2, MOVE_STEP * 5, MOVE_STEP * 10]:
        for dx, dy in directions:
            new_x = original_x + dx * distance
            new_y = original_y + dy * distance
            
            # Create test object with new position
            test_obj = RoomObject(
                id=obj.id,
                label=obj.label,
                bbox=[new_x, new_y, obj.width, obj.height],
                type=obj.type,
                orientation=obj.orientation,
                is_locked=obj.is_locked
            )
            
            # Check room bounds
            if not check_room_bounds(test_obj, room_width, room_height):
                continue
            
            # Check for overlaps with other objects
            has_overlap = False
            min_clearance = float('inf')
            
            for other in other_objects:
                if check_overlap(test_obj, other):
                    has_overlap = True
                    break
                clearance = calculate_clearance(test_obj, other)
                min_clearance = min(min_clearance, clearance)
            
            if has_overlap:
                continue
            
            # Score this position (prefer more clearance)
            position_score = min_clearance
            
            if position_score > best_score:
                best_score = position_score
                best_position = (new_x, new_y)
    
    return best_position


def generate_optimization_summary(state: AgentState) -> str:
    """Generate a summary of the optimization process."""
    initial = state.get("initial_score")
    current = state.get("current_score")
    iterations = state.get("iteration_count", 0)
    
    if not initial or not current:
        return "Optimization not yet complete."
    
    improvement = current.total_score - initial.total_score
    
    summary = f"""
=== Optimization Summary ===
Iterations: {iterations}
Initial Score: {initial.total_score:.1f}/100
Final Score: {current.total_score:.1f}/100
Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f} points

Breakdown:
  - Constraints: {current.constraint_score:.1f}/100
  - Walkability: {current.walkability_score:.1f}/100
  - Preferences: {current.preference_score:.1f}/100

{current.explanation}
"""
    return summary.strip()
