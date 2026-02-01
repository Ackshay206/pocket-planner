"""
Optimize Route

POST /optimize - Optimize a room layout while respecting locked objects.
This is the core endpoint that uses our LangGraph workflow.
"""

from fastapi import APIRouter, HTTPException

from app.models.api import OptimizeRequest, OptimizeResponse
from app.models.room import ConstraintViolation
from app.agents.graph import run_optimization
from app.core.constraints import check_all_hard_constraints


router = APIRouter(prefix="/optimize", tags=["Optimization"])


@router.post("", response_model=OptimizeResponse)
async def optimize_layout(request: OptimizeRequest) -> OptimizeResponse:
    """
    Optimize a room layout while respecting locked objects.
    
    This endpoint:
    1. Takes current layout and locked object IDs
    2. Runs the LangGraph optimization workflow
    3. Returns optimized layout with explanation
    
    The optimizer will:
    - Check all constraints (door clearance, overlaps, walking paths)
    - Move unlocked furniture to fix violations
    - Iterate until violations are resolved or max iterations reached
    - Generate human-readable explanation of changes
    """
    try:
        # Mark locked objects
        for obj in request.current_layout:
            if obj.id in request.locked_ids:
                obj.is_locked = True
        
        # Run the optimization workflow
        result = run_optimization(
            objects=request.current_layout,
            room_width=request.room_dimensions.width_estimate,
            room_height=request.room_dimensions.height_estimate,
            locked_ids=request.locked_ids,
            max_iterations=request.max_iterations
        )
        
        # Extract results
        new_layout = result.get("proposed_layout", request.current_layout)
        current_score = result.get("current_score")
        initial_score = result.get("initial_score")
        iterations = result.get("iteration_count", 0)
        explanation = result.get("explanation", "Optimization complete.")
        
        # Calculate improvement
        improvement = 0.0
        if current_score and initial_score:
            improvement = current_score.total_score - initial_score.total_score
        
        # Get remaining violations
        violations = check_all_hard_constraints(
            new_layout,
            request.room_dimensions.width_estimate,
            request.room_dimensions.height_estimate
        )
        
        return OptimizeResponse(
            new_layout=new_layout,
            explanation=explanation,
            layout_score=current_score.total_score if current_score else 0.0,
            iterations=iterations,
            constraint_violations=violations,
            improvement=round(improvement, 1)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/quick", response_model=OptimizeResponse)
async def quick_optimize(request: OptimizeRequest) -> OptimizeResponse:
    """
    Quick optimization with fewer iterations.
    
    Same as /optimize but limited to 2 iterations for faster response.
    """
    request.max_iterations = 2
    return await optimize_layout(request)
