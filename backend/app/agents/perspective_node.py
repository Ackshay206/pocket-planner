"""
Perspective Node

Generates photorealistic 3D perspective views of room layouts.
FULLY TRACED with LangSmith - including Gemini image generation calls.
"""

import base64
import asyncio
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from app.config import get_settings
from app.models.state import AgentState
from app.models.room import RoomObject, RoomDimensions

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_ENABLED = True
except ImportError:
    LANGSMITH_ENABLED = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# DEBUG HELPER
# ============================================================================
import os
import json
from datetime import datetime

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_logs")

def _ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)

def _save_debug_json(filename: str, data: Any):
    try:
        _ensure_debug_dir()
        filepath = os.path.join(DEBUG_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[Perspective] Saved debug: {filepath}")
    except Exception as e:
        print(f"[Perspective] Failed to save debug {filename}: {e}")

class PerspectiveGenerator:
    """
    Generates photorealistic perspective renders of room layouts.
    All methods are traced with LangSmith.
    """
    
    def __init__(self):
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=settings.google_api_key)
        self.image_model = settings.image_model_name


    @traceable(
        name="perspective_generator.generate_side_view", 
        run_type="chain", 
        tags=["perspective", "3d", "generation"]
    )
    async def generate_side_view(
        self,
        layout: List[RoomObject],
        room_dims: RoomDimensions,
        style: str = "modern",
        view_angle: str = "corner",
        lighting: str = "natural daylight",
        image_base64: Optional[str] = None,
        layout_plan: Optional[dict] = None,
    ) -> str:
        """
        Generate a photorealistic side/perspective view of the room.
        
        NOTE: image_base64 is passed to Gemini so it can see the actual
        room's visual identity (wall colors, flooring, furniture styles).
        The prompt strictly instructs it to change the camera angle from
        top-down to eye-level while preserving the room's appearance.
        
        TRACED: Full chain with Gemini image generation details.
        """
        # Build furniture descriptions from layout_plan if available,
        # otherwise fall back to bbox-based descriptions
        if layout_plan and layout_plan.get("furniture_placement"):
            furniture_lines = []
            for furn_id, placement_desc in layout_plan["furniture_placement"].items():
                furniture_lines.append(f"- {furn_id}: {placement_desc}")
            furniture_text = "\n".join(furniture_lines)
        else:
            furniture_descriptions = [
                self._describe_object(obj, room_dims)
                for obj in layout
                if obj.type.value == "movable"
            ]
            furniture_text = "\n".join(furniture_descriptions)
        
        # Build the generation prompt
        prompt = self._build_perspective_prompt(
            furniture_text, room_dims, style, view_angle, lighting
        )
        
        # Debug logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_debug_json(f"{timestamp}_perspective_INPUT.json", {
            "prompt": prompt,
            "layout_count": len(layout),
            "room_dims": room_dims.dict(),
            "style": style,
            "has_layout_plan": layout_plan is not None,
        })

        try:
            # Pass the layout thumbnail so Gemini can see the actual room
            # (wall colors, floor, windows, furniture styles).
            # The prompt strictly overrides the camera angle.
            result = await self._call_gemini_image_generation(prompt, image_base64)
            print(f"[Perspective] Generation successful")
            return result
        except Exception as e:
            _save_debug_json(f"{timestamp}_perspective_ERROR.json", {"error": str(e)})
            print(f"[Perspective] Generation failed: {e}")
            raise e

    @traceable(
        name="gemini_perspective_generation", 
        run_type="llm", 
        tags=["gemini", "image", "perspective", "api-call"],
        metadata={"model_type": "gemini-image", "task": "perspective_generation"}
    )
    async def _call_gemini_image_generation(self, prompt: str, image_base64: Optional[str] = None) -> str:
        """
        Make the Gemini image generation API call.
        TRACED as an LLM/image-gen call.
        """
        contents = [prompt]
        if image_base64:
             if "," in image_base64:
                 image_base64 = image_base64.split(",")[1]
             try:
                 image_data = base64.b64decode(image_base64)
                 contents.insert(0, types.Part.from_bytes(data=image_data, mime_type="image/png"))
             except Exception as e:
                 print(f"[Perspective] Failed to decode input image: {e}")

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.image_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"],
                temperature=0.3,
            )
        )
        
        # Extract image from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                return base64.b64encode(image_data).decode('utf-8')
        
        raise RuntimeError("No image generated in response")

    def _build_perspective_prompt(
        self,
        furniture_text: str,
        room_dims: RoomDimensions,
        style: str,
        view_angle: str,
        lighting: str
    ) -> str:
        """Build the prompt for perspective generation."""
        return f"""You are a professional architectural photographer. You have been given a 2D floor plan image as a REFERENCE for the room's appearance — its wall colors, flooring, window positions, and furniture styles.

Your job: produce a PHOTOREALISTIC PHOTOGRAPH of this SAME room, but shot from INSIDE the room at EYE LEVEL.

═══════════════════════════════════════════════════════════════
  ⛔ ABSOLUTE HARD RULES — VIOLATION = AUTOMATIC REJECTION
═══════════════════════════════════════════════════════════════
  1. The output MUST be a FIRST-PERSON EYE-LEVEL photograph.
  2. Camera height: exactly 5 feet (1.5 m) above the floor.
  3. Camera angle: HORIZONTAL. The lens faces STRAIGHT AHEAD,
     parallel to the floor. NOT tilted down, NOT looking down.
  4. You MUST see WALLS in the image — left wall, right wall,
     and the far wall. Walls MUST be visible and vertical.
  5. You MUST see the FLOOR in the bottom portion of the image,
     receding into the distance with perspective.
  6. You MUST see the CEILING (or ceiling edge) at the top.
  7. Furniture must appear IN PERSPECTIVE — items closer to the
     camera are larger, items farther away are smaller.
  8. There must be a clear VANISHING POINT in the image.

  ⛔ FORBIDDEN OUTPUTS (will be rejected):
  - Any top-down / bird's-eye / overhead / map view
  - Any isometric or axonometric projection
  - Any 2D floor plan or blueprint style
  - Any view where the camera looks DOWN at the floor
  - Any output that copies the camera angle of the input image
═══════════════════════════════════════════════════════════════

HOW TO USE THE INPUT IMAGE:
- The attached image is a 2D top-down floor plan.
- Use it ONLY to understand: wall colors, floor material, window
  locations, and what the furniture looks like.
- DO NOT copy its camera angle. The input is top-down. Your output
  MUST be eye-level. These are completely different viewpoints.
- Imagine you WALKED INTO this room and took a photo with your
  phone held at eye height. THAT is what you must generate.

ROOM:
- Dimensions: approximately {room_dims.width_estimate:.0f} x {room_dims.height_estimate:.0f} feet
- Ceiling height: 9 feet
- Style: {style}
- Lighting: {lighting}

FURNITURE POSITIONS:
{furniture_text}

CAMERA:
- Position: Standing at the {view_angle} entrance/doorway, one step inside
- Height: Eye-level (5 feet / 1.5m above floor)
- Direction: Looking across toward the opposite wall
- Lens: 35mm standard

REQUIRED IN THE OUTPUT:
✓ Vertical walls converging toward a vanishing point
✓ Floor in the lower third with perspective depth
✓ Ceiling edge visible at top
✓ 3D furniture with volume, shadows, materials
✓ Near objects larger, far objects smaller
✓ Realistic interior lighting

QUALITY: Photorealistic, professional interior design photography.

Generate the photograph now."""

    def _describe_object(
        self,
        obj: RoomObject,
        room_dims: RoomDimensions
    ) -> str:
        """Generate natural language description of object position."""
        x_pct = obj.bbox[0]
        y_pct = obj.bbox[1]
        
        # Determine position in room
        x_pos = "left" if x_pct < 33 else ("center" if x_pct < 66 else "right")
        y_pos = "front" if y_pct < 33 else ("middle" if y_pct < 66 else "back")
        
        # Orientation description
        orientation_map = {
            0: "facing north (away from viewer)", 
            90: "facing east (to the right)",
            180: "facing south (toward viewer)", 
            270: "facing west (to the left)"
        }
        orientation_desc = orientation_map.get(obj.orientation, "")
        
        # Material description
        material = f" made of {obj.material_hint}" if obj.material_hint else ""
        
        # Build description
        desc = f"- {obj.label.title()}{material} positioned in the {y_pos}-{x_pos} area of the room"
        
        if obj.label in ['bed', 'desk', 'sofa', 'chair'] and orientation_desc:
            desc += f", {orientation_desc}"
        
        return desc


# LangGraph node functions

@traceable(name="perspective_node", run_type="chain", tags=["langgraph", "node", "perspective"])
async def perspective_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node that generates perspective renders.
    TRACED: Full trace with image generation details.
    """
    generator = PerspectiveGenerator()
    
    try:
        layout = state.get("proposed_layout") or state.get("current_layout", [])
        room_dims = state["room_dimensions"]
        
        # Generate the main perspective view
        image_base64 = await generator.generate_side_view(
            layout=layout,
            room_dims=room_dims,
            style="modern",
            view_angle="corner",
            lighting="natural daylight"
        )
        
        return {
            "output_image_url": None,
            "output_image_base64": image_base64,
            "explanation": state.get("explanation", "") + "\n\nGenerated photorealistic perspective view.",
        }
        
    except Exception as e:
        return {
            "error": f"Perspective generation failed: {str(e)}",
            "output_image_base64": None
        }


def perspective_node_sync(state: AgentState) -> Dict[str, Any]:
    """Synchronous wrapper for LangGraph compatibility."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(perspective_node(state))
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, perspective_node(state))
            return future.result()