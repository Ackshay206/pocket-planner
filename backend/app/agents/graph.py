"""
LangGraph Workflow

Main agent workflow that orchestrates the optimization process:
VisionNode → ConstraintNode → SolverNode → (loop) → Output

Note: VisionNode and RenderNode are placeholders for Developer A's work.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from app.models.state import AgentState, create_initial_state
from app.models.room import RoomObject, RoomDimensions
from app.agents.constraint_node import constraint_node
from app.agents.solver_node import solver_node, generate_optimization_summary


# ============ Placeholder Nodes (Developer A will implement) ============

def vision_node(state: AgentState) -> dict:
    """
    Placeholder for Vision Node.
    
    Developer A will implement this to:
    1. Send image to Gemini 1.5 Pro Vision
    2. Extract room dimensions and objects
    3. Return structured JSON
    
    For now, we assume current_layout is already populated.
    """
    return {
        "explanation": "Vision analysis complete (placeholder)"
    }


def render_node(state: AgentState) -> dict:
    """
    Placeholder for Render Node.
    
    Developer A will implement this to:
    1. Generate image edit prompts
    2. Use Gemini to move furniture in the image
    3. Return the edited image URL
    
    For now, we just finalize the state.
    """
    summary = generate_optimization_summary(state)
    
    return {
        "proposed_layout": state["current_layout"],
        "explanation": summary,
        "output_image_url": None  # Will be set by Developer A
    }


# ============ Router Functions ============

def should_continue_optimization(state: AgentState) -> Literal["solver", "render"]:
    """
    Decide whether to continue optimizing or render the result.
    
    Returns:
        "solver" to continue optimization
        "render" to finalize and render
    """
    if state.get("should_continue", False):
        return "solver"
    return "render"


def check_for_errors(state: AgentState) -> Literal["constraint", "error"]:
    """Check if there's an error in the state."""
    if state.get("error"):
        return "error"
    return "constraint"


# ============ Graph Definition ============

def create_optimization_graph() -> StateGraph:
    """
    Create the LangGraph workflow for room optimization.
    
    Flow:
        START → vision → constraint → solver ←→ constraint → render → END
                                         ↓
                                    (loop until no violations or max iterations)
    """
    # Create graph with AgentState
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("vision", vision_node)
    graph.add_node("constraint", constraint_node)
    graph.add_node("solver", solver_node)
    graph.add_node("render", render_node)
    
    # Define edges
    graph.set_entry_point("vision")
    
    # vision → constraint
    graph.add_edge("vision", "constraint")
    
    # constraint → solver OR render (conditional)
    graph.add_conditional_edges(
        "constraint",
        should_continue_optimization,
        {
            "solver": "solver",
            "render": "render"
        }
    )
    
    # solver → constraint (loop back)
    graph.add_edge("solver", "constraint")
    
    # render → END
    graph.add_edge("render", END)
    
    return graph


def compile_graph():
    """Compile the optimization graph for execution."""
    graph = create_optimization_graph()
    return graph.compile()


# ============ Execution Helpers ============

def run_optimization(
    objects: list[RoomObject],
    room_width: int,
    room_height: int,
    locked_ids: list[str] = None,
    image_base64: str = "",
    max_iterations: int = 5
) -> AgentState:
    """
    Run the full optimization workflow.
    
    Args:
        objects: Detected room objects
        room_width: Room width in units
        room_height: Room height in units
        locked_ids: IDs of user-locked objects
        image_base64: Original room image (base64)
        max_iterations: Maximum optimization iterations
        
    Returns:
        Final AgentState with optimized layout
    """
    # Create initial state
    room_dims = RoomDimensions(
        width_estimate=room_width,
        height_estimate=room_height
    )
    
    initial_state = create_initial_state(
        image_base64=image_base64,
        room_dimensions=room_dims,
        objects=objects,
        locked_ids=locked_ids,
        max_iterations=max_iterations
    )
    
    # Compile and run graph
    app = compile_graph()
    
    # Execute
    final_state = app.invoke(initial_state)
    
    return final_state


def run_optimization_stream(
    objects: list[RoomObject],
    room_width: int,
    room_height: int,
    locked_ids: list[str] = None,
    image_base64: str = "",
    max_iterations: int = 5
):
    """
    Run optimization with streaming updates.
    
    Yields state updates as the workflow progresses.
    """
    room_dims = RoomDimensions(
        width_estimate=room_width,
        height_estimate=room_height
    )
    
    initial_state = create_initial_state(
        image_base64=image_base64,
        room_dimensions=room_dims,
        objects=objects,
        locked_ids=locked_ids,
        max_iterations=max_iterations
    )
    
    app = compile_graph()
    
    for step in app.stream(initial_state):
        yield step
