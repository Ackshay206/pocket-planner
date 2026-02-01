"""
Tests for Phase 1: Models and Geometry

Run with: pytest tests/test_phase1.py -v
"""

import sys
sys.path.insert(0, "f:/pocket-planner/backend")

from app.models.room import RoomObject, ObjectType, RoomDimensions, VisionOutput
from app.core.geometry import (
    bbox_to_polygon,
    check_overlap,
    calculate_clearance,
    is_path_blocked,
    find_collisions,
    calculate_furniture_density
)


# ============ Model Tests ============

def test_room_object_creation():
    """Test creating a basic RoomObject."""
    bed = RoomObject(
        id="bed_1",
        label="bed",
        bbox=[10, 10, 100, 200],
        type=ObjectType.MOVABLE
    )
    assert bed.id == "bed_1"
    assert bed.x == 10
    assert bed.y == 10
    assert bed.width == 100
    assert bed.height == 200
    assert bed.center == (60, 110)
    print("✓ RoomObject creation works")


def test_vision_output():
    """Test VisionOutput schema."""
    output = VisionOutput(
        room_dimensions=RoomDimensions(width_estimate=300, height_estimate=400),
        objects=[
            RoomObject(id="bed_1", label="bed", bbox=[10, 10, 100, 200]),
            RoomObject(id="door_1", label="door", bbox=[0, 150, 20, 80], type=ObjectType.STRUCTURAL)
        ]
    )
    assert len(output.objects) == 2
    assert output.room_dimensions.width_estimate == 300
    print("✓ VisionOutput schema works")


# ============ Geometry Tests ============

def test_bbox_to_polygon():
    """Test bounding box to polygon conversion."""
    poly = bbox_to_polygon([10, 10, 100, 50])
    assert poly.bounds == (10.0, 10.0, 110.0, 60.0)
    print("✓ bbox_to_polygon works")


def test_overlap_detection():
    """Test collision detection between objects."""
    bed = RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 200])
    desk_overlapping = RoomObject(id="desk_1", label="desk", bbox=[50, 50, 80, 40])
    desk_separate = RoomObject(id="desk_2", label="desk", bbox=[200, 200, 80, 40])
    
    assert check_overlap(bed, desk_overlapping) == True
    assert check_overlap(bed, desk_separate) == False
    print("✓ Overlap detection works")


def test_clearance_calculation():
    """Test distance calculation between objects."""
    bed = RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 100])
    desk = RoomObject(id="desk_1", label="desk", bbox=[150, 0, 50, 50])
    
    clearance = calculate_clearance(bed, desk)
    assert clearance == 50.0
    print("✓ Clearance calculation works")


def test_find_collisions():
    """Test finding all colliding objects."""
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 200]),
        RoomObject(id="desk_1", label="desk", bbox=[50, 50, 80, 40]),  # Overlaps bed
        RoomObject(id="chair_1", label="chair", bbox=[200, 200, 40, 40]),  # No overlap
    ]
    
    collisions = find_collisions(objects)
    assert len(collisions) == 1
    assert collisions[0][0] == "bed_1"
    assert collisions[0][1] == "desk_1"
    print("✓ find_collisions works")


def test_path_blocked():
    """Test walking path obstruction detection."""
    obstacles = [
        RoomObject(id="desk_1", label="desk", bbox=[100, 90, 80, 40]),  # In the path
    ]
    
    # Path from (0, 100) to (300, 100) should be blocked by desk
    blocked, blocker = is_path_blocked((0, 100), (300, 100), obstacles)
    assert blocked == True
    assert blocker == "desk_1"
    
    # Path from (0, 0) to (50, 0) should NOT be blocked
    blocked2, blocker2 = is_path_blocked((0, 0), (50, 0), obstacles)
    assert blocked2 == False
    print("✓ is_path_blocked works")


def test_furniture_density():
    """Test room occupancy calculation."""
    objects = [
        RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 200]),  # 20,000 sq units
    ]
    
    density = calculate_furniture_density(200, 200, objects)  # Room = 40,000 sq units
    assert density == 50.0  # 50% occupied
    print("✓ calculate_furniture_density works")


# ============ Run All Tests ============

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Phase 1 Tests: Models & Geometry")
    print("="*50 + "\n")
    
    test_room_object_creation()
    test_vision_output()
    test_bbox_to_polygon()
    test_overlap_detection()
    test_clearance_calculation()
    test_find_collisions()
    test_path_blocked()
    test_furniture_density()
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50 + "\n")
