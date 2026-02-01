"""
Constraint Node

LangGraph node that checks all constraints on the current layout
and populates constraint_violations in the state.
"""

from typing import Dict, Any

from app.models.state import AgentState
from app.models.room import RoomObject
from app.core.constraints import check_all_hard_constraints
from app.core.scoring import score_layout


def constraint_node(state: AgentState) -> Dict[str, Any]:
    """
    Check constraints on current layout.
    
    This node:
    1. Runs all hard constraint checks
    2. Calculates the current layout score
    3. Determines if optimization should continue
    
    Args:
        state: Current agent state
        
    Returns:
        State updates with violations and scores
    """
    current_layout = state["current_layout"]
    room_dims = state["room_dimensions"]
    
    # Check hard constraints
    violations = check_all_hard_constraints(
        current_layout,
        room_dims.width_estimate,
        room_dims.height_estimate
    )
    
    # Calculate current score
    current_score = score_layout(
        current_layout,
        room_dims.width_estimate,
        room_dims.height_estimate
    )
    
    # Set initial score on first iteration
    initial_score = state.get("initial_score")
    if initial_score is None:
        initial_score = current_score
    
    # Determine if we should continue
    iteration = state["iteration_count"]
    max_iter = state["max_iterations"]
    
    # Stop if: no violations, max iterations reached, or score is good enough
    should_continue = (
        len(violations) > 0 and 
        iteration < max_iter and
        current_score.total_score < 95
    )
    
    return {
        "constraint_violations": violations,
        "current_score": current_score,
        "initial_score": initial_score,
        "should_continue": should_continue
    }


def format_violations(state: AgentState) -> str:
    """Format constraint violations as human-readable text."""
    violations = state.get("constraint_violations", [])
    
    if not violations:
        return "No constraint violations detected."
    
    lines = [f"Found {len(violations)} issue(s):"]
    for v in violations:
        lines.append(f"  - {v.description}")
    
    return "\n".join(lines)
