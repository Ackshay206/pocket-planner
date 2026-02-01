"""
Tests for Phase 4: FastAPI Endpoints

Run with: pytest tests/test_phase4.py -v

Note: These tests use FastAPI's TestClient for in-process testing.
"""

import sys
sys.path.insert(0, "f:/pocket-planner/backend")

from fastapi.testclient import TestClient
from app.main import app


# Create test client
client = TestClient(app)


# ============ Health Check Tests ============

def test_root_endpoint():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    print("✓ Root endpoint works")


def test_health_endpoint():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("✓ Health endpoint works")


# ============ Analyze Endpoint Tests ============

def test_analyze_endpoint():
    """Test the POST /analyze endpoint."""
    response = client.post(
        "/api/v1/analyze",
        json={"image_base64": "test_image_data"}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "room_dimensions" in data
    assert "objects" in data
    assert len(data["objects"]) > 0
    assert "detected_issues" in data
    print(f"✓ Analyze endpoint works - detected {len(data['objects'])} objects")


def test_analyze_returns_mock_data():
    """Test that analyze returns properly structured mock data."""
    response = client.post(
        "/api/v1/analyze",
        json={"image_base64": "any_image"}
    )
    data = response.json()
    
    # Check structure of returned objects
    for obj in data["objects"]:
        assert "id" in obj
        assert "label" in obj
        assert "bbox" in obj
        assert len(obj["bbox"]) == 4
        assert "type" in obj
    
    # Check room dimensions
    dims = data["room_dimensions"]
    assert dims["width_estimate"] > 0
    assert dims["height_estimate"] > 0
    print("✓ Analyze returns properly structured data")


# ============ Optimize Endpoint Tests ============

def test_optimize_endpoint():
    """Test the POST /optimize endpoint."""
    request_body = {
        "current_layout": [
            {"id": "door_1", "label": "door", "bbox": [0, 150, 20, 80], "type": "structural"},
            {"id": "bed_1", "label": "bed", "bbox": [100, 200, 100, 150], "type": "movable"},
            {"id": "desk_1", "label": "desk", "bbox": [200, 50, 80, 50], "type": "movable"}
        ],
        "locked_ids": ["bed_1"],
        "room_dimensions": {"width_estimate": 300, "height_estimate": 400},
        "max_iterations": 2
    }
    
    response = client.post("/api/v1/optimize", json=request_body)
    assert response.status_code == 200
    data = response.json()
    
    assert "new_layout" in data
    assert "explanation" in data
    assert "layout_score" in data
    assert "iterations" in data
    print(f"✓ Optimize endpoint works - score: {data['layout_score']}/100")


def test_optimize_respects_locked_objects():
    """Test that optimization doesn't move locked objects."""
    original_bed_pos = [100, 200, 100, 150]
    
    request_body = {
        "current_layout": [
            {"id": "door_1", "label": "door", "bbox": [0, 150, 20, 80], "type": "structural"},
            {"id": "bed_1", "label": "bed", "bbox": original_bed_pos, "type": "movable"},
        ],
        "locked_ids": ["bed_1"],
        "room_dimensions": {"width_estimate": 300, "height_estimate": 400},
        "max_iterations": 3
    }
    
    response = client.post("/api/v1/optimize", json=request_body)
    data = response.json()
    
    # Find bed in new layout
    bed = next(obj for obj in data["new_layout"] if obj["id"] == "bed_1")
    
    # Bed position should be unchanged
    assert bed["bbox"] == original_bed_pos
    print("✓ Optimize respects locked objects")


def test_quick_optimize_endpoint():
    """Test the /optimize/quick endpoint."""
    request_body = {
        "current_layout": [
            {"id": "bed_1", "label": "bed", "bbox": [100, 100, 100, 200], "type": "movable"}
        ],
        "locked_ids": [],
        "room_dimensions": {"width_estimate": 300, "height_estimate": 400}
    }
    
    response = client.post("/api/v1/optimize/quick", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert data["iterations"] <= 2  # Quick mode limits to 2 iterations
    print("✓ Quick optimize endpoint works")


# ============ Render Endpoint Tests ============

def test_render_endpoint():
    """Test the POST /render endpoint."""
    request_body = {
        "original_image_base64": "original_image_data",
        "final_layout": [
            {"id": "desk_1", "label": "desk", "bbox": [150, 50, 80, 50], "type": "movable"}
        ],
        "original_layout": [
            {"id": "desk_1", "label": "desk", "bbox": [50, 50, 80, 50], "type": "movable"}
        ]
    }
    
    response = client.post("/api/v1/render", json=request_body)
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    # Should detect the desk moved
    assert "desk" in data["message"].lower()
    print("✓ Render endpoint works")


def test_render_no_changes():
    """Test render endpoint when no changes were made."""
    request_body = {
        "original_image_base64": "image_data",
        "final_layout": [
            {"id": "bed_1", "label": "bed", "bbox": [100, 100, 100, 200], "type": "movable"}
        ],
        "original_layout": [
            {"id": "bed_1", "label": "bed", "bbox": [100, 100, 100, 200], "type": "movable"}
        ]
    }
    
    response = client.post("/api/v1/render", json=request_body)
    data = response.json()
    
    assert "unchanged" in data["message"].lower()
    print("✓ Render correctly handles no changes")


# ============ Run All Tests ============

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Phase 4 Tests: FastAPI Endpoints")
    print("="*50 + "\n")
    
    # Health checks
    test_root_endpoint()
    test_health_endpoint()
    
    # Analyze
    test_analyze_endpoint()
    test_analyze_returns_mock_data()
    
    # Optimize
    test_optimize_endpoint()
    test_optimize_respects_locked_objects()
    test_quick_optimize_endpoint()
    
    # Render
    test_render_endpoint()
    test_render_no_changes()
    
    print("\n" + "="*50)
    print("✅ ALL PHASE 4 TESTS PASSED!")
    print("="*50 + "\n")
