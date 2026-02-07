"""
Vision Node

Handles room image analysis using Gemini Vision.

FIXES v3.1:
1. Use gemini-2.5-flash for vision (faster, avoids cold-start timeout)
2. FULL detailed bed vs sofa prompt (restored from v1)
3. Post-processing deduplication for bed/sofa
4. Window detection: explicit prompt instructions + light-pattern inference
5. VisionOutput includes image_width/image_height for pixel-space calculations
6. JSON fence stripping in parser
7. Sofa label normalization
"""

import json
import base64
import io
from typing import List, Optional
from PIL import Image
import functools
from google import genai
from google.genai import types

from app.config import get_settings
from app.models.room import RoomObject, RoomDimensions, ObjectType, VisionOutput

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_ENABLED = True
except ImportError:
    LANGSMITH_ENABLED = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_current_run_tree():
        return None


STRUCTURAL_OBJECTS = {
    "door", "window", "wall", "shower", "bathtub", "bath", "toilet",
    "sink", "stovetop", "stove", "oven", "refrigerator", "fridge",
    "built-in", "builtin", "radiator", "fireplace", "stairs", "staircase",
    "column", "pillar", "beam", "hvac", "vent", "ac_unit", "washer",
    "dryer", "dishwasher", "water_heater", "boiler", "furnace"
}


ROOM_ANALYSIS_PROMPT = """You are an expert architectural floor plan analyzer. Analyze this top-down 2D floor plan image with EXTREME PRECISION.

## YOUR TASK
Detect ALL objects in this floor plan and provide TIGHT, ACCURATE bounding boxes.

## CRITICAL BOUNDING BOX INSTRUCTIONS
- The bounding box format is [ymin, xmin, ymax, xmax] normalized to 0-1000 scale
- (0,0) is TOP-LEFT corner, (1000,1000) is BOTTOM-RIGHT corner
- BOUNDING BOXES MUST BE TIGHT - fit exactly around each object with minimal padding
- Each object should have its OWN separate bounding box - do not merge objects

## OBJECT CLASSIFICATION

### STRUCTURAL (type: "structural") - CANNOT be moved:
- Doors, windows, walls
- Plumbing fixtures: toilet, sink, shower, bathtub
- Kitchen appliances: refrigerator, stove, stovetop, oven, dishwasher
- Built-in elements: fireplace, radiator, stairs, columns
- HVAC: vents, AC units

### MOVABLE (type: "movable") - CAN be rearranged:
- Beds, nightstands, dressers, wardrobes
- Desks, chairs, office furniture
- Sofas, couches, armchairs, tables, rugs, lamps, artwork, plants

## ⚠️ WINDOW DETECTION — VERY IMPORTANT
Windows are CRITICAL for layout planning. Detect them even if subtle:
- Look for LIGHT RAYS / SUNLIGHT PATCHES on the floor — these indicate a window on the wall the light comes FROM
- Look for bright rectangular areas on walls
- Look for thin horizontal/vertical lines on walls that differ from the wall color
- If you see diagonal light/shadow patterns on the floor, trace them back to find the window
- In 3D-style floor plans, windows often appear as bright openings or glass panels on walls
- If you see sunlight streaming in from a wall, there MUST be a window on that wall — add it even if the window frame isn't clearly drawn
- Label windows as "window" with type "structural"

## ⚠️ CRITICAL: DISTINGUISHING BEDS vs SOFAS
This is VERY important — beds and sofas are often confused in floor plans. Use these rules:

### BED characteristics:
- Usually the LARGEST furniture piece in a bedroom
- Typically placed with the HEAD (short side) against a wall
- Has a roughly 2:3 or 3:4 width-to-length ASPECT RATIO (e.g. twin ~40x75in, queen ~60x80in, king ~76x80in)
- Often has NIGHTSTANDS on one or both long sides
- Usually has PILLOWS drawn at one short end
- Located in a BEDROOM (private room, often with a closet/wardrobe nearby)
- A room should almost NEVER have more than ONE bed (unless it's a shared/kids room)

### SOFA / COUCH characteristics:
- Typically NARROWER and more ELONGATED than a bed (higher length-to-depth ratio)
- Depth is usually 30-40 inches while length varies (60-100+ inches)
- Often placed against a wall in a LIVING AREA or open space
- Commonly faces a TV, coffee table, or the center of a seating arrangement
- May have an L-SHAPE or SECTIONAL form
- Usually has a COFFEE TABLE or RUG in front of it
- NOT typically accompanied by nightstands

### Decision rule:
- If only ONE large rectangular item exists in a bedroom → label it "bed"
- If there is ALREADY a bed in the room and another large upholstered item exists → the second one is likely a "sofa"
- If the item is in a living room / open area / faces a TV → label it "sofa"
- If the aspect ratio is very elongated (depth < 40% of length) → likely "sofa"
- When in doubt, consider the CONTEXT: what other furniture is nearby?

## DETECTION GUIDELINES
1. Scan the ENTIRE image systematically from top-left to bottom-right
2. Identify room boundaries (walls) first
3. Detect EVERY piece of furniture, no matter how small
4. Give each object a unique ID: "{label}_{number}" (e.g., "bed_1", "window_1")
5. Pay special attention to: nightstands, chairs, rugs, artwork, AND WINDOWS
6. Do NOT label two items as "bed" unless clearly a shared room
7. Do NOT label multiple items as "bed" unless you are highly confident the room has multiple sleeping areas

## OUTPUT FORMAT
Return ONLY valid JSON:
{
    "room_dimensions": {
        "width_estimate": <room width in feet>,
        "height_estimate": <room height in feet>
    },
    "wall_bounds": [ymin, xmin, ymax, xmax],
    "objects": [
        {
            "id": "<label>_<number>",
            "label": "<object_type>",
            "box_2d": [ymin, xmin, ymax, xmax],
            "type": "movable" | "structural"
        }
    ]
}

IMPORTANT: Be thorough. Better to detect too many than too few.
ALWAYS detect windows — if you see light/sun patches, there is a window.
"""


class VisionAgent:
    def __init__(self):
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=settings.google_api_key)
        # Use flash model — much faster, avoids cold-start timeouts
        self.model = settings.model_name

    @traceable(name="vision_agent.analyze_room", run_type="chain", tags=["vision", "gemini"])
    async def analyze_room(self, image_base64: str) -> VisionOutput:
        image = self._decode_base64_image(image_base64)
        image_width, image_height = image.size

        response_text = await self._call_gemini_vision(image, image_width, image_height)
        vision_output = self._parse_gemini_response(response_text, image_width, image_height)
        vision_output.objects = self._deduplicate_beds_and_sofas(vision_output.objects)

        # Attach pixel dimensions so designer_node can do spatial math correctly
        vision_output.image_width = image_width
        vision_output.image_height = image_height

        return vision_output

    @traceable(name="gemini_vision_call", run_type="llm", tags=["gemini", "vision", "api-call"])
    async def _call_gemini_vision(self, image: Image.Image, width: int, height: int) -> str:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, ROOM_ANALYSIS_PROMPT],
            config=config
        )
        return response.text

    def _decode_base64_image(self, image_base64: str) -> Image.Image:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data))

    def _convert_box_2d_to_bbox(self, box_2d: List[int], image_width: int, image_height: int) -> List[int]:
        ymin, xmin, ymax, xmax = box_2d
        ymin = max(0, min(1000, ymin))
        xmin = max(0, min(1000, xmin))
        ymax = max(0, min(1000, ymax))
        xmax = max(0, min(1000, xmax))

        x = int(xmin / 1000 * image_width)
        y = int(ymin / 1000 * image_height)
        width = int((xmax - xmin) / 1000 * image_width)
        height = int((ymax - ymin) / 1000 * image_height)
        return [x, y, max(1, width), max(1, height)]

    def _is_structural_object(self, label: str) -> bool:
        label_lower = label.lower().replace(" ", "_").replace("-", "_")
        if label_lower in STRUCTURAL_OBJECTS:
            return True
        for structural in STRUCTURAL_OBJECTS:
            if structural in label_lower or label_lower in structural:
                return True
        return False

    def _deduplicate_beds_and_sofas(self, objects: List[RoomObject]) -> List[RoomObject]:
        beds = [obj for obj in objects if obj.label == "bed"]
        if len(beds) <= 1:
            return objects

        beds_with_area = []
        for bed in beds:
            w, h = bed.bbox[2], bed.bbox[3]
            area = w * h
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1
            beds_with_area.append((bed, area, aspect))
        beds_with_area.sort(key=lambda x: x[1], reverse=True)
        primary_bed = beds_with_area[0]

        updated = []
        for obj in objects:
            if obj.label == "bed" and obj.id != primary_bed[0].id:
                w, h = obj.bbox[2], obj.bbox[3]
                area = w * h
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1
                if area < primary_bed[1] * 0.6 or aspect < 0.4 or len(beds) > 2:
                    new_id = obj.id.replace("bed", "sofa")
                    updated.append(RoomObject(
                        id=new_id, label="sofa", bbox=obj.bbox.copy(),
                        type=obj.type, orientation=obj.orientation,
                        is_locked=obj.is_locked, z_index=obj.z_index,
                        material_hint=obj.material_hint,
                    ))
                    print(f"[Vision] Reclassified '{obj.id}' bed → sofa")
                    continue
            updated.append(obj)
        return updated

    @traceable(name="parse_vision_response", run_type="parser", tags=["parsing"])
    def _parse_gemini_response(self, response_text: str, image_width: int, image_height: int) -> VisionOutput:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")

        dims = data.get("room_dimensions", {})
        room_dimensions = RoomDimensions(
            width_estimate=dims.get("width_estimate", image_width),
            height_estimate=dims.get("height_estimate", image_height)
        )

        wall_bounds = None
        if "wall_bounds" in data and data["wall_bounds"]:
            wall_bounds = self._convert_box_2d_to_bbox(data["wall_bounds"], image_width, image_height)

        objects = []
        for obj_data in data.get("objects", []):
            label = obj_data.get("label", "unknown").lower()
            if label in ("couch", "loveseat", "settee", "divan"):
                label = "sofa"

            if self._is_structural_object(label):
                obj_type = ObjectType.STRUCTURAL
            elif obj_data.get("type") == "structural":
                obj_type = ObjectType.STRUCTURAL
            else:
                obj_type = ObjectType.MOVABLE

            box_2d = obj_data.get("box_2d", [0, 0, 100, 100])
            bbox = self._convert_box_2d_to_bbox(box_2d, image_width, image_height)
            objects.append(RoomObject(
                id=obj_data.get("id", f"obj_{len(objects)}"),
                label=label, bbox=bbox, type=obj_type, orientation=0
            ))

        return VisionOutput(
            room_dimensions=room_dimensions,
            objects=objects,
            wall_bounds=wall_bounds
        )


@functools.lru_cache()
def get_vision_agent() -> VisionAgent:
    return VisionAgent()