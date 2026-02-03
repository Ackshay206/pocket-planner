"""
Vision Node

Integrates with Gemini 2.5 Flash for vision extraction.
Uses the Gemini Developer API with simple API key authentication.
"""

import os
import json
import base64
from typing import Optional
from google import genai
from google.genai import types
import asyncio
from app.models.api import AnalyzeResponse
from app.models.room import RoomObject, RoomDimensions, ObjectType


class VisionExtractor:
    """
    Extracts room objects from bedroom images using Gemini Vision.
    """
    
    def __init__(self):
        # Using Gemini Developer API with API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"  # Vision model
    
    async def extract_objects(self, image_base64: str, max_retries: int = 3) -> AnalyzeResponse:
        """
        Extract room objects from an image using Gemini Vision.
        
        Args:
            image_base64: Base64 encoded image string
            max_retries: Number of retry attempts on failure
            
        Returns:
            AnalyzeResponse with detected objects and room dimensions
        """
        prompt = """Analyze this bedroom image and extract all objects. Return JSON with exact schema:
{
  "room_dimensions": {"width": float, "height": float},
  "objects": [
    {
      "id": "label_index",
      "label": "bed|desk|chair|door|window|wall|dresser|nightstand",
      "bbox": [x_percent, y_percent, width_percent, height_percent],
      "type": "movable|structural",
      "orientation": "north|south|east|west|null"
    }
  ]
}
Estimate room dimensions in feet based on typical furniture sizes.
Bounding boxes should be percentages (0-100) of image dimensions."""

        # Handle data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        image_data = base64.b64decode(image_base64)
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                data = json.loads(response.text)
                
                # Convert to RoomObjects
                objects = []
                for obj in data.get("objects", []):
                    obj_type = obj.get("type", "movable")
                    room_obj = RoomObject(
                        id=obj.get("id", f"obj_{len(objects)}"),
                        label=obj.get("label", "unknown"),
                        bbox=[int(b) for b in obj.get("bbox", [0, 0, 10, 10])],
                        type=ObjectType.STRUCTURAL if obj_type == "structural" else ObjectType.MOVABLE,
                        orientation=0,
                        is_locked=False
                    )
                    objects.append(room_obj)
                
                dims = data.get("room_dimensions", {"width": 300, "height": 400})
                return AnalyzeResponse(
                    room_dimensions=RoomDimensions(
                        width_estimate=int(dims.get("width", 300)),
                        height_estimate=int(dims.get("height", 400))
                    ),
                    objects=objects
                )
                
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Vision extraction failed after {max_retries} attempts: {e}")
        
        raise RuntimeError("Vision extraction failed")
