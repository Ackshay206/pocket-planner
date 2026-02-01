"""
Tests for Phase 2: Constraints, Scoring, Room Graph

Run with: pytest tests/test_phase2.py -v
"""

import sys
sys.path.insert(0, "f:/pocket-planner/backend")

from app.models.room import RoomObject, ObjectType
from app.core.constraints import (
    check_door_clearance,
    check_no_overlap,
    check_all_hard_constraints,
    evaluate_soft_constraints
)
from app.core.scoring import (
    score_layout,
    calculate_constraint_score,
    calculate_walkability_score
)
from app.core.room_graph import RoomGraph, EdgeRelation


# ============ Constraint Tests ============

def test_door_clearance_violation():
    """Test detection of furniture blocking door."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[25, 100, 100, 200], type=ObjectType.MOVABLE),  # Too close!
    ]
    
    violations = check_door_clearance(objects, min_clearance=60)
    assert len(violations) > 0
    assert "door_clearance" in violations[0].constraint_name
    print("✓ Door clearance violation detected")


def test_door_clearance_ok():
    """Test no violation when door has sufficient clearance."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[150, 100, 100, 200], type=ObjectType.MOVABLE),  # Far enough
    ]
    
    violations = check_door_clearance(objects, min_clearance=60)
    assert len(violations) == 0
    print("✓ Door clearance OK when sufficient")


def test_no_overlap_violation():
    """Test detection of overlapping furniture."""
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 200]),
        RoomObject(id="desk_1", label="desk", bbox=[50, 50, 80, 40]),  # Overlaps bed
    ]
    
    violations = check_no_overlap(objects)
    assert len(violations) == 1
    assert "bed_1" in violations[0].objects_involved
    assert "desk_1" in violations[0].objects_involved
    print("✓ Overlap violation detected")


def test_all_hard_constraints():
    """Test combined hard constraint checking."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[50, 50, 100, 200]),
        RoomObject(id="desk_1", label="desk", bbox=[50, 100, 80, 40]),  # Overlaps bed
    ]
    
    violations = check_all_hard_constraints(objects, room_width=300, room_height=400)
    assert len(violations) >= 1  # At least overlap
    print("✓ All hard constraints checked")


def test_soft_constraints():
    """Test soft constraint evaluation."""
    objects = [
        RoomObject(id="window_1", label="window", bbox=[200, 0, 50, 20], type=ObjectType.STRUCTURAL),
        RoomObject(id="desk_1", label="desk", bbox=[180, 30, 60, 40]),  # Near window
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[100, 200, 100, 150]),  # Away from door
    ]
    
    score, suggestions = evaluate_soft_constraints(objects)
    assert score > 50  # Should be reasonably good
    print(f"✓ Soft constraints evaluated: score={score:.1f}")


# ============ Scoring Tests ============

def test_perfect_layout_score():
    """Test scoring of a well-optimized layout."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 150, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="window_1", label="window", bbox=[250, 0, 50, 20], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[150, 200, 100, 150]),  # Away from door
        RoomObject(id="desk_1", label="desk", bbox=[200, 30, 80, 50]),   # Near window
    ]
    
    result = score_layout(objects, room_width=300, room_height=400)
    assert result.total_score > 60
    print(f"✓ Layout scored: {result.total_score}/100")


def test_bad_layout_score():
    """Test scoring of a poor layout with violations."""
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[10, 100, 100, 200]),  # Blocking door & overlapping
        RoomObject(id="desk_1", label="desk", bbox=[50, 150, 80, 40]),  # Overlapping bed
    ]
    
    result = score_layout(objects, room_width=300, room_height=400)
    assert result.constraint_score < 80  # Should have violations
    print(f"✓ Bad layout penalized: {result.total_score}/100")


# ============ Room Graph Tests ============

def test_graph_creation():
    """Test creating a room graph."""
    graph = RoomGraph()
    
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[100, 100, 100, 200]),
        RoomObject(id="nightstand_1", label="nightstand", bbox=[200, 150, 30, 30]),
    ]
    
    graph.add_objects(objects)
    assert len(graph) == 2
    print("✓ Room graph created")


def test_object_locking():
    """Test locking/unlocking objects."""
    graph = RoomGraph()
    
    bed = RoomObject(id="bed_1", label="bed", bbox=[100, 100, 100, 200])
    graph.add_object(bed)
    
    assert graph.lock_object("bed_1") == True
    assert "bed_1" in graph.get_locked_objects()
    
    assert graph.unlock_object("bed_1") == True
    assert "bed_1" not in graph.get_locked_objects()
    print("✓ Object locking works")


def test_dependency_detection():
    """Test automatic dependency detection."""
    graph = RoomGraph()
    
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[100, 100, 100, 200]),
        RoomObject(id="nightstand_1", label="nightstand", bbox=[205, 150, 30, 30]),  # Close to bed
    ]
    
    graph.add_objects(objects)
    
    # Nightstand should depend on bed
    deps = graph.get_dependencies("nightstand_1")
    assert "bed_1" in deps
    print("✓ Dependencies detected")


def test_movable_objects():
    """Test getting movable (unlocked, non-structural) objects."""
    graph = RoomGraph()
    
    objects = [
        RoomObject(id="door_1", label="door", bbox=[0, 100, 20, 80], type=ObjectType.STRUCTURAL),
        RoomObject(id="bed_1", label="bed", bbox=[100, 100, 100, 200], type=ObjectType.MOVABLE),
        RoomObject(id="desk_1", label="desk", bbox=[200, 50, 80, 50], type=ObjectType.MOVABLE),
    ]
    
    graph.add_objects(objects)
    graph.lock_object("bed_1")
    
    movable = graph.get_movable_objects()
    assert "desk_1" in movable
    assert "bed_1" not in movable  # Locked
    assert "door_1" not in movable  # Structural
    print("✓ Movable objects filtered correctly")


# ============ Run All Tests ============

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Phase 2 Tests: Constraints, Scoring, Room Graph")
    print("="*50 + "\n")
    
    # Constraint tests
    test_door_clearance_violation()
    test_door_clearance_ok()
    test_no_overlap_violation()
    test_all_hard_constraints()
    test_soft_constraints()
    
    # Scoring tests
    test_perfect_layout_score()
    test_bad_layout_score()
    
    # Room graph tests
    test_graph_creation()
    test_object_locking()
    test_dependency_detection()
    test_movable_objects()
    
    print("\n" + "="*50)
    print("✅ ALL PHASE 2 TESTS PASSED!")
    print("="*50 + "\n")
