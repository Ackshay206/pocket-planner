"""
Tests for Phase 3: Agent Nodes and LangGraph Workflow

Run with: pytest tests/test_phase3.py -v
"""

import sys
sys.path.insert(0, "f:/pocket-planner/backend")

from app.models.room import RoomObject, ObjectType, RoomDimensions
from app.models.state import AgentState, create_initial_state
from app.agents.constraint_node import constraint_node
from app.agents.solver_node import solver_node, find_better_position
from app.agents.graph import run_optimization, compile_graph


# ============ State Tests ============

def test_create_initial_state():
    """Test creating initial agent state."""
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[100, 100, 100, 200]),
        RoomObject(id="desk_1", label="desk", bbox=[200, 50, 80, 50]),
    ]
    
    state = create_initial_state(
        image_base64="test_image",
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=objects,
        locked_ids=["bed_1"],
        max_iterations=3
    )
    
    assert state["iteration_count"] == 0
    assert len(state["current_layout"]) == 2
    assert "bed_1" in state["locked_object_ids"]
    assert state["should_continue"] == True
    print("✓ Initial state created correctly")


# ============ Constraint Node Tests ============

def test_constraint_node_finds_violations():
    """Test that constraint node detects violations."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[25, 100, 100, 200]),  # Blocking door!
    ]
    
    state = create_initial_state(
        image_base64="",
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=objects
    )
    
    result = constraint_node(state)
    
    assert len(result["constraint_violations"]) > 0
    assert result["current_score"] is not None
    assert result["current_score"].total_score < 80  # Should be penalized
    print(f"✓ Constraint node found {len(result['constraint_violations'])} violation(s)")


def test_constraint_node_clean_layout():
    """Test constraint node with a good layout."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[150, 200, 100, 150]),  # Far from door
    ]
    
    state = create_initial_state(
        image_base64="",
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=objects
    )
    
    result = constraint_node(state)
    
    # Should have few or no violations
    assert result["current_score"].total_score > 60
    print(f"✓ Clean layout scored: {result['current_score'].total_score:.1f}/100")


# ============ Solver Node Tests ============

def test_find_better_position():
    """Test finding a better position for an object."""
    obj = RoomObject(id="desk_1", label="desk", bbox=[10, 100, 80, 50])  # Too close to edge
    
    other_objects = [
        RoomObject(id="bed_1", label="bed", bbox=[150, 50, 100, 200]),
    ]
    
    new_pos = find_better_position(
        obj=obj,
        all_objects=[obj] + other_objects,
        room_width=300,
        room_height=400,
        locked_ids=set()
    )
    
    # Should find a position with more clearance
    assert new_pos is not None
    print(f"✓ Found better position: {new_pos}")


def test_solver_node_makes_changes():
    """Test that solver node attempts to fix violations."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="desk_1", label="desk", bbox=[25, 100, 80, 50]),  # Blocking door!
    ]
    
    state = create_initial_state(
        image_base64="",
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=objects,
        locked_ids=[]
    )
    
    # First run constraint node to get violations
    constraint_result = constraint_node(state)
    state.update(constraint_result)
    
    # Then run solver
    solver_result = solver_node(state)
    
    assert solver_result["iteration_count"] == 1
    assert "explanation" in solver_result
    print(f"✓ Solver made changes: {solver_result['explanation'][:50]}...")


def test_solver_respects_locked_objects():
    """Test that solver doesn't move locked objects."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[30, 100, 100, 200], is_locked=True),  # Locked!
    ]
    
    state = create_initial_state(
        image_base64="",
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=objects,
        locked_ids=["bed_1"]
    )
    
    # Run constraint and solver
    constraint_result = constraint_node(state)
    state.update(constraint_result)
    solver_result = solver_node(state)
    
    # Bed's position should not change
    bed = next(o for o in solver_result["current_layout"] if o.id == "bed_1")
    assert bed.bbox[0] == 30  # X unchanged
    assert bed.bbox[1] == 100  # Y unchanged
    print("✓ Solver respects locked objects")


# ============ Full Workflow Tests ============

def test_graph_compilation():
    """Test that the LangGraph compiles without errors."""
    app = compile_graph()
    assert app is not None
    print("✓ Graph compiled successfully")


def test_full_optimization_workflow():
    """Test the complete optimization workflow."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 150, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="window_1", label="window", bbox=[250, 0, 50, 20], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[100, 50, 100, 200]),
        RoomObject(id="desk_1", label="desk", bbox=[50, 150, 80, 50]),  # Near door - suboptimal
    ]
    
    result = run_optimization(
        objects=objects,
        room_width=300,
        room_height=400,
        locked_ids=["bed_1"],  # Lock the bed
        max_iterations=3
    )
    
    assert result is not None
    assert result.get("proposed_layout") is not None
    assert result.get("explanation") != ""
    
    # Check that bed wasn't moved (it's locked)
    bed = next(o for o in result["proposed_layout"] if o.id == "bed_1")
    assert bed.bbox[0] == 100
    
    print(f"✓ Full workflow completed in {result['iteration_count']} iteration(s)")
    print(f"  Final score: {result['current_score'].total_score:.1f}/100")


# ============ Run All Tests ============

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Phase 3 Tests: Agent Nodes & LangGraph Workflow")
    print("="*50 + "\n")
    
    # State tests
    test_create_initial_state()
    
    # Constraint node tests
    test_constraint_node_finds_violations()
    test_constraint_node_clean_layout()
    
    # Solver node tests
    test_find_better_position()
    test_solver_node_makes_changes()
    test_solver_respects_locked_objects()
    
    # Workflow tests
    test_graph_compilation()
    test_full_optimization_workflow()
    
    print("\n" + "="*50)
    print("✅ ALL PHASE 3 TESTS PASSED!")
    print("="*50 + "\n")
