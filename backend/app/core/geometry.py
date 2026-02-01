"""
Geometry Utilities

Shapely-based functions for spatial operations:
- Converting bounding boxes to polygons
- Collision/overlap detection
- Clearance (distance) calculations
- Path blocking detection
"""

from typing import List, Tuple, Optional
from shapely.geometry import Polygon, box, LineString, Point
from shapely.ops import nearest_points

from app.models.room import RoomObject


def bbox_to_polygon(bbox: List[int]) -> Polygon:
    """
    Convert a bounding box [x, y, width, height] to a Shapely Polygon.
    
    Args:
        bbox: [x, y, width, height] where (x,y) is top-left corner
        
    Returns:
        Shapely Polygon representing the bounding box
        
    Example:
        >>> poly = bbox_to_polygon([10, 10, 100, 50])
        >>> poly.bounds
        (10.0, 10.0, 110.0, 60.0)
    """
    x, y, w, h = bbox
    return box(x, y, x + w, y + h)


def object_to_polygon(obj: RoomObject) -> Polygon:
    """Convert a RoomObject to a Shapely Polygon."""
    return bbox_to_polygon(obj.bbox)


def check_overlap(obj_a: RoomObject, obj_b: RoomObject) -> bool:
    """
    Check if two objects overlap (collide).
    
    Args:
        obj_a: First room object
        obj_b: Second room object
        
    Returns:
        True if objects overlap, False otherwise
        
    Example:
        >>> bed = RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 200])
        >>> desk = RoomObject(id="desk_1", label="desk", bbox=[50, 50, 80, 40])
        >>> check_overlap(bed, desk)
        True
    """
    poly_a = object_to_polygon(obj_a)
    poly_b = object_to_polygon(obj_b)
    return poly_a.intersects(poly_b)


def calculate_overlap_area(obj_a: RoomObject, obj_b: RoomObject) -> float:
    """
    Calculate the overlapping area between two objects.
    
    Returns:
        Overlap area in square units. Returns 0 if no overlap.
    """
    poly_a = object_to_polygon(obj_a)
    poly_b = object_to_polygon(obj_b)
    intersection = poly_a.intersection(poly_b)
    return intersection.area


def calculate_clearance(obj_a: RoomObject, obj_b: RoomObject) -> float:
    """
    Calculate the minimum distance (clearance) between two objects.
    
    Args:
        obj_a: First room object
        obj_b: Second room object
        
    Returns:
        Distance in units. Returns 0 if objects overlap.
        
    Example:
        >>> bed = RoomObject(id="bed_1", label="bed", bbox=[0, 0, 100, 100])
        >>> desk = RoomObject(id="desk_1", label="desk", bbox=[150, 0, 50, 50])
        >>> calculate_clearance(bed, desk)
        50.0
    """
    poly_a = object_to_polygon(obj_a)
    poly_b = object_to_polygon(obj_b)
    return poly_a.distance(poly_b)


def get_buffered_polygon(obj: RoomObject, buffer_distance: float) -> Polygon:
    """
    Create a polygon with a buffer zone around the object.
    
    Useful for checking clearance requirements (e.g., door swing area).
    
    Args:
        obj: Room object
        buffer_distance: Distance to expand in all directions
        
    Returns:
        Buffered Shapely Polygon
    """
    poly = object_to_polygon(obj)
    return poly.buffer(buffer_distance)


def is_path_blocked(
    start: Tuple[int, int],
    end: Tuple[int, int],
    obstacles: List[RoomObject],
    path_width: float = 45.0
) -> Tuple[bool, Optional[str]]:
    """
    Check if a walking path between two points is blocked by obstacles.
    
    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        obstacles: List of objects that could block the path
        path_width: Required path width in units (default 45cm)
        
    Returns:
        Tuple of (is_blocked, blocking_object_id or None)
        
    Example:
        >>> door_center = (10, 200)
        >>> bed_center = (150, 200)
        >>> obstacles = [desk_obj, chair_obj]
        >>> blocked, blocker = is_path_blocked(door_center, bed_center, obstacles)
    """
    # Create a line representing the walking path
    path_line = LineString([start, end])
    
    # Buffer the path to account for required walking width
    path_corridor = path_line.buffer(path_width / 2)
    
    for obj in obstacles:
        # Skip structural elements that are doorways
        if obj.type.value == "structural" and obj.label == "door":
            continue
            
        poly = object_to_polygon(obj)
        if path_corridor.intersects(poly):
            return (True, obj.id)
    
    return (False, None)


def find_collisions(objects: List[RoomObject]) -> List[Tuple[str, str, float]]:
    """
    Find all pairs of overlapping objects.
    
    Args:
        objects: List of all room objects
        
    Returns:
        List of tuples: (obj_a_id, obj_b_id, overlap_area)
    """
    collisions = []
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1:]:
            overlap = calculate_overlap_area(obj_a, obj_b)
            if overlap > 0:
                collisions.append((obj_a.id, obj_b.id, overlap))
    return collisions


def check_room_bounds(
    obj: RoomObject,
    room_width: int,
    room_height: int
) -> bool:
    """
    Check if an object is within the room boundaries.
    
    Returns:
        True if object is fully within room bounds
    """
    return (
        obj.x >= 0 and
        obj.y >= 0 and
        obj.x + obj.width <= room_width and
        obj.y + obj.height <= room_height
    )


def get_free_space(
    room_width: int,
    room_height: int,
    objects: List[RoomObject]
) -> Polygon:
    """
    Calculate the free (unoccupied) space in the room.
    
    Returns:
        Polygon representing available floor space
    """
    room = box(0, 0, room_width, room_height)
    
    for obj in objects:
        poly = object_to_polygon(obj)
        room = room.difference(poly)
    
    return room


def calculate_furniture_density(
    room_width: int,
    room_height: int,
    objects: List[RoomObject]
) -> float:
    """
    Calculate what percentage of the room is occupied by furniture.
    
    Returns:
        Percentage (0-100) of room area occupied
    """
    room_area = room_width * room_height
    if room_area == 0:
        return 0.0
    
    furniture_area = sum(obj.width * obj.height for obj in objects)
    return (furniture_area / room_area) * 100
