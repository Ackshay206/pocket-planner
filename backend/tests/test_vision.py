"""
Tests for Vision Service

Run with: pytest tests/test_vision.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from app.core.vision_service import (
    convert_box_2d_to_bbox,
    parse_gemini_response,
    decode_base64_image
)
from app.models.room import ObjectType


# ============ Unit Tests (No API calls) ============

def test_convert_box_2d_to_bbox():
    """Test conversion from Gemini [ymin,xmin,ymax,xmax] to our [x,y,w,h] format."""
    # Gemini returns normalized 0-1000 coords: [ymin, xmin, ymax, xmax]
    box_2d = [100, 200, 300, 400]  # 10%-30% y, 20%-40% x
    
    # For a 1000x1000 image
    bbox = convert_box_2d_to_bbox(box_2d, image_width=1000, image_height=1000)
    
    # Expected: x=200, y=100, width=200, height=200
    assert bbox[0] == 200  # x
    assert bbox[1] == 100  # y
    assert bbox[2] == 200  # width (400-200)
    assert bbox[3] == 200  # height (300-100)
    print("✓ Box conversion works")


def test_convert_box_2d_to_bbox_different_image_size():
    """Test conversion with non-square image."""
    box_2d = [0, 0, 500, 500]  # Half the normalized space
    
    # For a 800x600 image
    bbox = convert_box_2d_to_bbox(box_2d, image_width=800, image_height=600)
    
    assert bbox[0] == 0      # x
    assert bbox[1] == 0      # y
    assert bbox[2] == 400    # width (500/1000 * 800)
    assert bbox[3] == 300    # height (500/1000 * 600)
    print("✓ Box conversion works for non-square images")


def test_parse_gemini_response_valid():
    """Test parsing a valid Gemini JSON response."""
    response_text = '''{
        "room_dimensions": {"width_estimate": 400, "height_estimate": 300},
        "objects": [
            {"id": "bed_1", "label": "bed", "box_2d": [200, 100, 600, 500], "type": "movable"},
            {"id": "door_1", "label": "door", "box_2d": [0, 0, 200, 100], "type": "structural"}
        ]
    }'''
    
    result = parse_gemini_response(response_text, image_width=800, image_height=600)
    
    assert result.room_dimensions.width_estimate == 400
    assert result.room_dimensions.height_estimate == 300
    assert len(result.objects) == 2
    
    bed = result.objects[0]
    assert bed.id == "bed_1"
    assert bed.label == "bed"
    assert bed.type == ObjectType.MOVABLE
    
    door = result.objects[1]
    assert door.type == ObjectType.STRUCTURAL
    print("✓ Gemini response parsing works")


def test_parse_gemini_response_invalid_json():
    """Test handling of invalid JSON response."""
    with pytest.raises(ValueError):
        parse_gemini_response("not valid json", 800, 600)
    print("✓ Invalid JSON handled correctly")


def test_parse_gemini_response_empty_objects():
    """Test handling response with no objects detected."""
    response_text = '''{
        "room_dimensions": {"width_estimate": 300, "height_estimate": 400},
        "objects": []
    }'''
    
    result = parse_gemini_response(response_text, 800, 600)
    assert len(result.objects) == 0
    print("✓ Empty objects handled correctly")


# ============ Run All Tests ============

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Vision Service Tests")
    print("="*50 + "\n")
    
    test_convert_box_2d_to_bbox()
    test_convert_box_2d_to_bbox_different_image_size()
    test_parse_gemini_response_valid()
    test_parse_gemini_response_empty_objects()
    
    # This one uses pytest.raises, run separately
    try:
        test_parse_gemini_response_invalid_json()
    except SystemExit:
        pass
    
    print("\n" + "="*50)
    print("✅ ALL VISION TESTS PASSED!")
    print("="*50 + "\n")
