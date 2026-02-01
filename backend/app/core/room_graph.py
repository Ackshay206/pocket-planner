"""
Room Graph

NetworkX-based graph for managing spatial relationships between objects.
Tracks which objects depend on each other for positioning.
"""

from typing import List, Dict, Optional, Set
from enum import Enum
import networkx as nx

from app.models.room import RoomObject, ObjectType
from app.core.geometry import calculate_clearance


class EdgeRelation(str, Enum):
    """Types of relationships between room objects."""
    ADJACENT = "adjacent"       # Objects are next to each other (< 20 units)
    NEAR = "near"               # Within proximity threshold (< 50 units)
    BLOCKING = "blocking"       # One object blocks access to another
    DEPENDS_ON = "depends_on"   # One object's position depends on another


# Proximity thresholds
ADJACENT_THRESHOLD = 20.0
NEAR_THRESHOLD = 50.0


class RoomGraph:
    """
    Graph-based representation of room object relationships.
    
    Nodes: Room objects (furniture and structural elements)
    Edges: Spatial relationships between objects
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._locked_ids: Set[str] = set()
    
    def add_object(self, obj: RoomObject) -> None:
        """Add a room object as a node in the graph."""
        self.graph.add_node(
            obj.id,
            label=obj.label,
            type=obj.type.value,
            bbox=obj.bbox,
            is_locked=obj.is_locked
        )
        if obj.is_locked:
            self._locked_ids.add(obj.id)
    
    def add_objects(self, objects: List[RoomObject]) -> None:
        """Add multiple objects and automatically detect relationships."""
        for obj in objects:
            self.add_object(obj)
        
        # Detect relationships between all pairs
        self._detect_relationships(objects)
    
    def _detect_relationships(self, objects: List[RoomObject]) -> None:
        """Automatically detect spatial relationships between objects."""
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1:]:
                distance = calculate_clearance(obj_a, obj_b)
                
                if distance < ADJACENT_THRESHOLD:
                    self.add_relationship(obj_a.id, obj_b.id, EdgeRelation.ADJACENT)
                elif distance < NEAR_THRESHOLD:
                    self.add_relationship(obj_a.id, obj_b.id, EdgeRelation.NEAR)
        
        # Add semantic dependencies
        self._add_semantic_dependencies(objects)
    
    def _add_semantic_dependencies(self, objects: List[RoomObject]) -> None:
        """Add logical dependencies (e.g., nightstand depends on bed)."""
        beds = [o for o in objects if o.label == "bed"]
        nightstands = [o for o in objects if o.label == "nightstand"]
        
        # Nightstands should be near beds
        for nightstand in nightstands:
            if beds:
                # Find nearest bed
                nearest_bed = min(beds, key=lambda b: calculate_clearance(nightstand, b))
                self.add_relationship(
                    nightstand.id, 
                    nearest_bed.id, 
                    EdgeRelation.DEPENDS_ON
                )
    
    def add_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        relation: EdgeRelation
    ) -> None:
        """Add a directed relationship between two objects."""
        self.graph.add_edge(from_id, to_id, relation=relation.value)
    
    def lock_object(self, object_id: str) -> bool:
        """
        Lock an object so it cannot be moved.
        
        Returns:
            True if successfully locked, False if object doesn't exist
        """
        if object_id in self.graph.nodes:
            self.graph.nodes[object_id]['is_locked'] = True
            self._locked_ids.add(object_id)
            return True
        return False
    
    def unlock_object(self, object_id: str) -> bool:
        """Unlock an object so it can be moved."""
        if object_id in self.graph.nodes:
            self.graph.nodes[object_id]['is_locked'] = False
            self._locked_ids.discard(object_id)
            return True
        return False
    
    def get_locked_objects(self) -> Set[str]:
        """Get set of all locked object IDs."""
        return self._locked_ids.copy()
    
    def get_movable_objects(self) -> List[str]:
        """Get list of object IDs that can be moved."""
        return [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('type') == 'movable' and not data.get('is_locked', False)
        ]
    
    def get_structural_objects(self) -> List[str]:
        """Get list of structural (fixed) object IDs."""
        return [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('type') == 'structural'
        ]
    
    def get_dependencies(self, object_id: str) -> List[str]:
        """Get objects that this object depends on."""
        deps = []
        for _, target, data in self.graph.out_edges(object_id, data=True):
            if data.get('relation') == EdgeRelation.DEPENDS_ON.value:
                deps.append(target)
        return deps
    
    def get_dependents(self, object_id: str) -> List[str]:
        """Get objects that depend on this object."""
        deps = []
        for source, _, data in self.graph.in_edges(object_id, data=True):
            if data.get('relation') == EdgeRelation.DEPENDS_ON.value:
                deps.append(source)
        return deps
    
    def get_adjacent_objects(self, object_id: str) -> List[str]:
        """Get objects adjacent to this object."""
        adjacent = []
        for _, target, data in self.graph.out_edges(object_id, data=True):
            if data.get('relation') == EdgeRelation.ADJACENT.value:
                adjacent.append(target)
        for source, _, data in self.graph.in_edges(object_id, data=True):
            if data.get('relation') == EdgeRelation.ADJACENT.value:
                adjacent.append(source)
        return list(set(adjacent))
    
    def to_dict(self) -> Dict:
        """Export graph as dictionary for serialization."""
        return {
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": [
                {"from": u, "to": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
            "locked_ids": list(self._locked_ids)
        }
    
    def get_optimization_order(self) -> List[str]:
        """
        Get order in which objects should be optimized.
        
        Objects with dependencies should be moved after their dependencies.
        """
        # Use topological sort for dependency order
        try:
            order = list(nx.topological_sort(self.graph))
            # Filter to only movable, unlocked objects
            return [
                obj_id for obj_id in order
                if obj_id in self.get_movable_objects()
            ]
        except nx.NetworkXUnfeasible:
            # Cycle detected, return movable objects in any order
            return self.get_movable_objects()
    
    def __len__(self) -> int:
        return len(self.graph.nodes)
    
    def __repr__(self) -> str:
        return f"RoomGraph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"
