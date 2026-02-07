"""
Designer Node - Hierarchical Zone-Based Layout Generation

ALL FIXES CONSOLIDATED:
1. _extract_element_info uses PIXEL bbox space, not room-feet space
2. _prepare_objects computes pixel extent from bboxes
3. Window fallback: infer opposite door if not detected
4. Specs reference "table/desk" for rooms without a desk
5. Reinforcement auto-generated from technical_spec (not hardcoded)
6. Anti-duplication language in image prompt
7. Exclusion zones: structural fixtures → no-go areas
8. Post-plan validation against structural objects
9. Wall objects filtered from structural list
10. Debug logging for all inputs/outputs
"""

import json
import base64
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from google import genai
from google.genai import types

from app.config import get_settings
from app.models.state import AgentState
from app.models.room import RoomObject, RoomDimensions, ObjectType
from app.core.scoring import score_layout

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
# DEBUG
# ============================================================================
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_logs")

def _ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)

def _save_debug_json(filename: str, data: Any):
    try:
        _ensure_debug_dir()
        filepath = os.path.join(DEBUG_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[DEBUG] Saved: {filepath}")
    except Exception as e:
        print(f"[DEBUG] Failed to save {filename}: {e}")

def _save_debug_image(filename: str, image_base64: str):
    try:
        _ensure_debug_dir()
        filepath = os.path.join(DEBUG_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_base64))
        print(f"[DEBUG] Saved image: {filepath}")
    except Exception as e:
        print(f"[DEBUG] Failed to save image {filename}: {e}")

# ============================================================================
# ZONES
# ============================================================================
class ZoneType(str, Enum):
    WORK = "work_zone"
    SLEEP = "sleep_zone"
    LIVING = "living_zone"

FURNITURE_ZONE_MAP = {
    "bed": ZoneType.SLEEP, "nightstand": ZoneType.SLEEP, "dresser": ZoneType.SLEEP,
    "wardrobe": ZoneType.SLEEP, "closet": ZoneType.SLEEP,
    "desk": ZoneType.WORK, "office_chair": ZoneType.WORK,
    "bookshelf": ZoneType.WORK, "filing_cabinet": ZoneType.WORK,
    "sofa": ZoneType.LIVING, "couch": ZoneType.LIVING, "armchair": ZoneType.LIVING,
    "coffee_table": ZoneType.LIVING, "dining_table": ZoneType.LIVING,
    "tv_stand": ZoneType.LIVING, "rug": ZoneType.LIVING, "plant": ZoneType.LIVING,
    "lamp": ZoneType.LIVING,
}

LAYOUT_SPECIFICATIONS = {
    "work_focused": {
        "name": "Productivity Focus",
        "zone_priorities": [ZoneType.WORK, ZoneType.SLEEP, ZoneType.LIVING],
        "description": "A work-first setup. The table/desk is the HERO piece, placed in the prime position. The bed is hidden in a far corner.",
        "technical_spec": {
            "desk_rule": "CRITICAL: The Table/Desk MUST be pushed flush against the wall that has the window. If no window, place it on the wall OPPOSITE the door so it's the first thing visible on entry. The table/desk is the HERO piece.",
            "desk_orientation": "Position the Table/Desk so that it is on the straight line from the door to the opposite wall. Chairs go between the table and room center.",
            "bed_hide": "CRITICAL: The Bed MUST be in the corner FURTHEST from both the door AND the table/desk. Headboard flush against wall. Bed should NOT be the first thing you see entering.",
            "bed_rotation": "If the bed is currently near the table/desk or near the door, it MUST be relocated to a different wall entirely.",
            "nightstand_rule": "Nightstands follow the bed to its NEW position.",
            "sofa_rule": "Sofa flat against a side wall that is NOT the table/desk wall and NOT the bed wall. Keep center path door→table clear.",
            "sightline": "When someone opens the door, the FIRST thing they should see is the Table/Desk and work area, NOT the bed.",
            "pathway": "Clear straight walking path from door to table/desk."
        }
    },
    "cozy": {
        "name": "Cozy Retreat",
        "zone_priorities": [ZoneType.SLEEP, ZoneType.LIVING, ZoneType.WORK],
        "description": "Comfort-first. The bed and sofa form one unified relaxation block.",
        "technical_spec": {
            "bed_focal": "CRITICAL: The Bed must be centered on the wall OPPOSITE the door — it's the first thing you see entering.",
            "bed_anchor": "Center the Bed with one nightstand on each side (left AND right).",
            "sofa_at_foot": "CRITICAL: The Sofa MUST be at the FOOT of the bed (the end away from the headboard), facing the bed. Bed + sofa form ONE unified lounging block.",
            "table_hide": "Tuck the Table/Desk into a far corner, out of the main sightline from the door.",
            "sofa_direction": "Sofa faces INWARD toward the bed, NOT toward a wall."
        }
    },
    "space_optimized": {
        "name": "Space Optimized",
        "zone_priorities": [ZoneType.LIVING, ZoneType.SLEEP, ZoneType.WORK],
        "description": "Perimeter layout. ALL furniture hugs the walls. The center floor is EMPTY.",
        "technical_spec": {
            "perimeter_rule": "CRITICAL: ALL furniture (Bed, Sofa, Table, everything) MUST be pushed flat against walls. Nothing floating.",
            "center_void": "CRITICAL: The physical center of the room must be COMPLETELY EMPTY floor space. No furniture, no rugs in the middle.",
            "sofa_long_wall": "Sofa along a long wall, never floating.",
            "table_wall": "Table/Desk against a wall, preferably near window if available.",
            "pathway": "Wide clear walking path from door to all zones.",
            "door_strict": "NO furniture within 3 feet of the door."
        }
    }
}


class InteriorDesignerAgent:
    def __init__(self):
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = settings.model_name
        self.image_model = settings.image_model_name

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    @traceable(name="designer_agent.generate_layout_variations", run_type="chain", tags=["designer"])
    async def generate_layout_variations(
        self, current_layout: List[RoomObject], room_dims: RoomDimensions,
        locked_ids: List[str], image_base64: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        self._debug_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        max_x = max((obj.bbox[0] + obj.bbox[2] for obj in current_layout), default=600)
        max_y = max((obj.bbox[1] + obj.bbox[3] for obj in current_layout), default=700)
        self._pixel_width = max(max_x + 50, 400)
        self._pixel_height = max(max_y + 50, 400)

        complete_locked_ids, movable_objects, structural_objects, door_info, window_info = \
            self._prepare_objects(current_layout, locked_ids)

        if not movable_objects:
            raise ValueError("No movable objects to arrange")

        zone_assignments = self._classify_furniture_to_zones(movable_objects)
        movable_count = len(movable_objects)
        furniture_labels = [obj["label"] for obj in movable_objects]

        print(f"[Designer] Movable ({movable_count}): {furniture_labels}")
        print(f"[Designer] Structural ({len(structural_objects)}): {[o['label'] for o in structural_objects]}")
        print(f"[Designer] Door: {door_info}")
        print(f"[Designer] Window: {window_info}")
        print(f"[Designer] Pixel space: {self._pixel_width}x{self._pixel_height}")

        valid_ids = {obj["id"] for obj in movable_objects}
        for zone in zone_assignments:
            zone_assignments[zone] = [oid for oid in zone_assignments[zone] if oid in valid_ids]

        _save_debug_json(f"{self._debug_ts}_00_shared_context.json", {
            "pixel_space": {"width": self._pixel_width, "height": self._pixel_height},
            "room_dims_feet": {"width": room_dims.width_estimate, "height": room_dims.height_estimate},
            "door_info": door_info, "window_info": window_info,
            "movable_objects": movable_objects, "structural_objects": structural_objects,
        })

        if image_base64:
            clean_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
            _save_debug_image(f"{self._debug_ts}_00_input_image.jpg", clean_b64)

        # STEP 1: Generate plans in parallel (with image for visual context)
        plan_tasks = [
            self._generate_layout_plan(sk, sp, zone_assignments, movable_objects,
                structural_objects, room_dims, door_info, window_info, image_base64)
            for sk, sp in LAYOUT_SPECIFICATIONS.items()
        ]
        layout_plans = await asyncio.gather(*plan_tasks, return_exceptions=True)

        # STEP 2: Validate and refine plans against actual room image
        validation_tasks = []
        plan_indices = []
        for i, plan in enumerate(layout_plans):
            if isinstance(plan, Exception):
                continue
            sk = list(LAYOUT_SPECIFICATIONS.keys())[i]
            sp = LAYOUT_SPECIFICATIONS[sk]
            validation_tasks.append(
                self._validate_layout_compliance(image_base64, plan, sp, sk)
            )
            plan_indices.append(i)

        validated_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # STEP 3: Generate images from validated plans
        image_tasks, valid_plans = [], []
        for vi, orig_i in enumerate(plan_indices):
            sk = list(LAYOUT_SPECIFICATIONS.keys())[orig_i]
            sp = LAYOUT_SPECIFICATIONS[sk]

            validated_plan = validated_results[vi]
            if isinstance(validated_plan, Exception):
                print(f"[Designer] Validation failed for {sk}: {validated_plan}")
                validated_plan = layout_plans[orig_i]  # Fallback to original plan

            # Filter hallucinated IDs
            if validated_plan and "furniture_placement" in validated_plan:
                validated_plan["furniture_placement"] = {
                    k: v for k, v in validated_plan["furniture_placement"].items() if k in valid_ids
                }

            _save_debug_json(f"{self._debug_ts}_plan_{sk}_VALIDATED.json", {"plan": validated_plan})

            if validated_plan and image_base64:
                valid_plans.append((sk, sp, validated_plan))
                image_tasks.append(self._generate_layout_image(
                    validated_plan, sk, sp, movable_objects, structural_objects,
                    door_info, window_info, image_base64, movable_count, furniture_labels
                ))

        if not image_tasks:
            raise ValueError("No valid layout plans generated")

        images = await asyncio.gather(*image_tasks, return_exceptions=True)

        variations = []
        for i, (sk, sp, plan) in enumerate(valid_plans):
            img = images[i] if i < len(images) else None
            if isinstance(img, Exception):
                print(f"[Designer] Image failed {sk}: {img}")
                continue
            if img:
                _save_debug_image(f"{self._debug_ts}_image_{sk}_OUTPUT.png", img)
            variations.append({
                "name": sp["name"], "style_key": sk,
                "description": plan.get("description", sp["description"]),
                "layout": current_layout, "layout_plan": plan,
                "thumbnail_base64": img,
            })

        if not variations:
            raise ValueError("Failed to generate any valid layouts")
        return variations

    # ========================================================================
    # PREPARE OBJECTS
    # ========================================================================
    def _prepare_objects(
        self, current_layout: List[RoomObject], locked_ids: List[str]
    ) -> Tuple[Set[str], List[dict], List[dict], Optional[Dict], Optional[Dict]]:

        complete_locked = set(locked_ids)
        for obj in current_layout:
            if obj.type == ObjectType.STRUCTURAL or obj.is_locked:
                complete_locked.add(obj.id)

        movable, structural = [], []
        all_doors, all_windows = [], []
        pw, ph = self._pixel_width, self._pixel_height

        for obj in current_layout:
            clean_label = obj.label.lower().replace("_", " ").split("_")[0]
            obj_dict = {
                "id": obj.id, "label": clean_label, "full_label": obj.label,
                "bbox": obj.bbox.copy() if isinstance(obj.bbox, list) else list(obj.bbox),
            }

            if obj.id in complete_locked:
                structural.append(obj_dict)
                if "door" in obj.label.lower():
                    all_doors.append(self._extract_element_info(obj, pw, ph, "door"))
                if "window" in obj.label.lower():
                    all_windows.append(self._extract_element_info(obj, pw, ph, "window"))
            else:
                movable.append(obj_dict)

        door_info = all_doors[0] if all_doors else None
        window_info = all_windows[0] if all_windows else None

        # Fallback: infer window opposite door
        if not window_info and door_info:
            opposite = {"north (top)": "south (bottom)", "south (bottom)": "north (top)",
                        "east (right)": "west (left)", "west (left)": "east (right)"}
            inferred_wall = opposite.get(door_info["wall"], "west (left)")
            window_info = {
                "id": "inferred_window", "type": "window", "wall": inferred_wall,
                "inferred": True, "x": 0, "y": 0, "width": 0, "height": 0,
                "center_x": 0, "center_y": 0, "position_on_wall_percent": 50.0,
            }
            print(f"[Designer] No window detected — inferred on {inferred_wall} (opposite door)")

        # Filter walls from structural list (visible in image, just adds noise)
        structural_for_designer = [o for o in structural if o["label"] not in ("wall",)]
        filtered = len(structural) - len(structural_for_designer)
        if filtered > 0:
            print(f"[Designer] Filtered {filtered} wall objects from structural list")

        return complete_locked, movable, structural_for_designer, door_info, window_info

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    def _extract_element_info(self, obj: RoomObject, pw: int, ph: int, element_type: str) -> Dict:
        """Wall detection using PIXEL coordinates."""
        bbox = obj.bbox if isinstance(obj.bbox, list) else list(obj.bbox)
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        wall, pct = "interior", 50.0
        margin_x, margin_y = pw * 0.15, ph * 0.15

        dist_top, dist_bot = y, ph - (y + h)
        dist_left, dist_right = x, pw - (x + w)
        min_dist = min(dist_top, dist_bot, dist_left, dist_right)

        if min_dist == dist_top and dist_top < margin_y:
            wall, pct = "north (top)", cx / pw * 100
        elif min_dist == dist_bot and dist_bot < margin_y:
            wall, pct = "south (bottom)", cx / pw * 100
        elif min_dist == dist_left and dist_left < margin_x:
            wall, pct = "west (left)", cy / ph * 100
        elif min_dist == dist_right and dist_right < margin_x:
            wall, pct = "east (right)", cy / ph * 100

        return {"id": obj.id, "type": element_type, "wall": wall,
                "bbox": [x, y, w, h], "x": x, "y": y, "width": w, "height": h,
                "center_x": cx, "center_y": cy, "position_on_wall_percent": round(pct, 1)}

    def _classify_furniture_to_zones(self, movable_objects: List[dict]) -> Dict[ZoneType, List[str]]:
        zones = {ZoneType.WORK: [], ZoneType.SLEEP: [], ZoneType.LIVING: []}
        for obj in movable_objects:
            label = obj["label"].lower()
            assigned = False
            for key, zone in FURNITURE_ZONE_MAP.items():
                if key in label:
                    zones[zone].append(obj["id"])
                    assigned = True
                    break
            if not assigned:
                if "chair" in label:
                    zones[ZoneType.WORK].append(obj["id"])
                else:
                    zones[ZoneType.LIVING].append(obj["id"])
        return zones

    def _build_exclusion_zones(self, structural_objects: List[dict]) -> str:
        """Convert structural bboxes into no-go zone descriptions."""
        zones = []
        pw, ph = self._pixel_width, self._pixel_height
        for obj in structural_objects:
            label = obj["label"]
            if label in ("window", "door", "wall", "shelf", "cabinet"):
                continue
            bbox = obj["bbox"]
            x, y, w, h = bbox
            cx, cy = x + w / 2, y + h / 2
            x_pos = "left (west)" if cx < pw * 0.33 else ("center" if cx < pw * 0.66 else "right (east)")
            y_pos = "top (north)" if cy < ph * 0.33 else ("middle" if cy < ph * 0.66 else "bottom (south)")
            area_pct = (w * h) / (pw * ph) * 100
            if area_pct < 0.5:
                continue
            zones.append(f"  - {label} ({obj['id']}): fixed at {y_pos}-{x_pos}. NO furniture here.")
        if not zones:
            return ""
        return "⚠️ EXCLUSION ZONES — furniture MUST NOT overlap these fixed fixtures:\n" + "\n".join(zones)

    def _validate_plan_against_structures(self, plan: Dict, structural_objects: List[dict], style_key: str) -> List[str]:
        """Log warnings if placements might overlap structural objects."""
        warnings = []
        pw, ph = self._pixel_width, self._pixel_height
        floor_fixtures = {}
        for obj in structural_objects:
            if obj["label"] in ("window", "door", "wall", "shelf", "cabinet"):
                continue
            bbox = obj["bbox"]
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            x_z = "west" if cx < pw * 0.33 else ("center" if cx < pw * 0.66 else "east")
            y_z = "north" if cy < ph * 0.33 else ("center" if cy < ph * 0.66 else "south")
            floor_fixtures[obj["id"]] = {"label": obj["label"], "zone": f"{y_z}-{x_z}"}

        for furn_id, desc in plan.get("furniture_placement", {}).items():
            desc_lower = desc.lower()
            for struct_id, info in floor_fixtures.items():
                if info["label"] in desc_lower or struct_id in desc_lower:
                    warnings.append(f"[{style_key}] '{furn_id}' references '{info['label']}' ({struct_id}) — potential overlap")
        for w in warnings:
            print(f"[Designer WARNING] {w}")
        return warnings

    def _build_reinforcement(self, style_key: str, spec: Dict, window_wall: str, door_wall: str) -> str:
        """Auto-generate image reinforcement from the technical_spec.
        This ensures the image prompt always matches the spec exactly."""
        tech = spec.get("technical_spec", {})
        lines = [f"⚠️ {spec['name'].upper()} — KEY RULES FOR IMAGE GENERATION:"]
        for key, value in tech.items():
            # Replace generic wall references with actual detected walls
            rule = value.replace("{window_wall}", window_wall).replace("{door_wall}", door_wall)
            lines.append(f"- {rule}")
        lines.append("- The output MUST look visibly DIFFERENT from the input image.")
        return "\n".join(lines)

    # ========================================================================
    # VALIDATION — refine plan against actual room image
    # ========================================================================
    @traceable(name="validate_layout_compliance", run_type="llm", tags=["gemini", "validation"])
    async def _validate_layout_compliance(
        self, image_base64: str, layout_plan: Dict, layout_spec: Dict, style_key: str
    ) -> Dict:
        """Validate and refine layout plan using the actual room image."""
        if not image_base64:
            return layout_plan

        constraints = layout_spec.get("technical_spec", {})
        constraints_text = "\n".join([f"- {v}" for v in constraints.values()]) if constraints else "No specific constraints."

        prompt = f"""You are an Expert Interior Design Validator.

## GOAL
Verify if the proposed furniture layout matches the designated STYLE in this specific room.
If the placement instructions are vague, generic, or violate the style, REWRITE them to be specific and correct.

## ROOM IMAGE
(Attached)

## STYLE GOAL: "{layout_spec['name']}"
{layout_spec['description']}

## STRICT CONSTRAINTS (MUST VERIFY):
{constraints_text}

## PROPOSED PLAN
{json.dumps(layout_plan.get('furniture_placement', {}), indent=2)}

## YOUR TASK
1. Look at the room image. Identify constraints (doors, windows, odd corners, kitchen, bathroom).
2. Check if the "PROPOSED PLAN" actually achieves the "{layout_spec['name']}" goal in THIS specific room.
3. If a furniture item's position is not ideal for this style, change it.
4. If a furniture item would overlap a structural fixture (toilet, shower, sink, stove), move it.
5. If a furniture item is blocking a door/path, move it.
6. Return the REFINED placement dictionary.

## OUTPUT FORMAT (JSON)
{{
  "furniture_placement": {{
    "furniture_id": "refined specific instruction...",
    ...
  }},
  "changes_made": ["list of changes and reasoning..."]
}}"""

        contents = [prompt]
        clean_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
        try:
            image_data = base64.b64decode(clean_b64)
            contents.insert(0, types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))
        except Exception as e:
            print(f"[Designer] Validation image decode failed: {e}")
            return layout_plan

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content, model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            data = json.loads(response.text)

            # Merge refined placements — only update existing items, don't drop any
            refined = data.get("furniture_placement", {})
            original = layout_plan.get("furniture_placement", {})
            for k, v in refined.items():
                if k in original:
                    original[k] = v
            layout_plan["furniture_placement"] = original

            if data.get("changes_made"):
                print(f"[Designer VALIDATOR] Changes for {style_key}: {data['changes_made']}")
                _save_debug_json(f"{self._debug_ts}_plan_{style_key}_VALIDATION_CHANGES.json", {
                    "changes": data["changes_made"], "refined_placements": refined
                })

            return layout_plan
        except Exception as e:
            print(f"[Designer] Validation failed for {style_key}: {e}")
            return layout_plan

    # ========================================================================
    # PLAN GENERATION — now receives image for visual context
    # ========================================================================
    @traceable(name="generate_layout_plan", run_type="llm", tags=["gemini", "planning"])
    async def _generate_layout_plan(
        self, style_key, spec, zone_assignments, movable_objects,
        structural_objects, room_dims, door_info, window_info, image_base64=None
    ) -> Dict[str, Any]:

        obj_lookup = {o["id"]: o for o in movable_objects}
        zone_furniture = {}
        for zt, ids in zone_assignments.items():
            zone_furniture[zt.value] = [{"id": i, "label": obj_lookup[i]["label"]} for i in ids if i in obj_lookup]

        constraints = spec.get("technical_spec", {})
        constraints_text = "\n".join([f"- {v}" for v in constraints.values()]) if constraints else "No specific constraints."

        door_desc = f"on the {door_info['wall']} wall at ~{door_info.get('position_on_wall_percent', 50):.0f}% along that wall" if door_info else "location unknown"
        if window_info and window_info.get("inferred"):
            window_desc = f"INFERRED on the {window_info['wall']} wall (assume this is where light comes from)"
        elif window_info:
            window_desc = f"on the {window_info['wall']} wall at ~{window_info.get('position_on_wall_percent', 50):.0f}% along that wall"
        else:
            window_desc = "not detected"

        exclusion_text = self._build_exclusion_zones(structural_objects)

        prompt = f"""You are an expert interior designer creating a "{spec['name']}" layout.

## ROOM INFO
- Room dimensions: ~{room_dims.width_estimate:.0f} x {room_dims.height_estimate:.0f} feet
- Door: {door_desc}
- Window: {window_desc}

## STYLE: {spec['name']}
{spec['description']}

## SPECIFIC CONSTRAINTS (MUST FOLLOW):
{constraints_text}

## FURNITURE TO ARRANGE (by zone)
{json.dumps(zone_furniture, indent=2)}

## STRUCTURAL ELEMENTS (DO NOT MOVE)
{json.dumps(structural_objects, indent=2)}

{exclusion_text}

## YOUR TASK
Create a layout plan using RELATIVE/SEMANTIC positions only.
Describe WHERE each piece goes relative to walls, other furniture, and structural elements.

## CRITICAL RULES
1. DOOR CLEARANCE: Nothing may block the door.
2. Include ALL furniture — do not skip any.
3. Follow the SPECIFIC CONSTRAINTS exactly — they are non-negotiable.
4. Do not add any new furniture.
5. Use the ACTUAL furniture labels from the list above (e.g. "table_1" not "desk").
6. Do NOT place any movable furniture where it would overlap a fixed fixture (toilet, shower, sink, stove, refrigerator).

## OUTPUT FORMAT (JSON)
{{
  "description": "2-3 sentences explaining the design rationale",
  "furniture_placement": {{
    "<furniture_id>": "<relative position description>",
    ...
  }},
  "door_clearance": "how door area is kept clear",
  "zone_arrangement": {{
    "work_zone": "location description",
    "sleep_zone": "location description",
    "living_zone": "location description"
  }}
}}"""

        _save_debug_json(f"{self._debug_ts}_plan_{style_key}_INPUT.json", {
            "style_key": style_key, "full_prompt": prompt,
            "door_info": door_info, "window_info": window_info,
        })

        # Build contents: image (if available) + text prompt
        contents = [prompt]
        if image_base64:
            clean_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
            try:
                img_data = base64.b64decode(clean_b64)
                contents.insert(0, types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
            except Exception as e:
                print(f"[Designer] Failed to decode image for plan {style_key}: {e}")

        response = await asyncio.to_thread(
            self.client.models.generate_content, model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
        )
        result = json.loads(response.text)
        self._validate_plan_against_structures(result, structural_objects, style_key)
        _save_debug_json(f"{self._debug_ts}_plan_{style_key}_OUTPUT.json", {"plan": result})
        return result

    # ========================================================================
    # IMAGE GENERATION
    # ========================================================================
    @traceable(name="generate_layout_image", run_type="llm", tags=["gemini", "image"])
    async def _generate_layout_image(
        self, layout_plan, style_key, spec, movable_objects, structural_objects,
        door_info, window_info, image_base64, movable_count, furniture_labels
    ) -> Optional[str]:

        placement = layout_plan.get("furniture_placement", {})
        placement_text = "\n".join([f"  • {k.upper()}: {v}" for k, v in placement.items()]) or "Follow zone arrangement"
        zone_arr = layout_plan.get("zone_arrangement", {})
        zone_text = "\n".join([f"  • {z}: {loc}" for z, loc in zone_arr.items()])

        door_wall = door_info['wall'] if door_info else "unknown"
        door_pct = f"at ~{door_info.get('position_on_wall_percent', 50):.0f}%" if door_info else ""
        window_wall = window_info['wall'] if window_info else "unknown"
        window_pct = f"at ~{window_info.get('position_on_wall_percent', 50):.0f}%" if window_info else ""

        furniture_list_str = ", ".join(furniture_labels)
        structural_list = ", ".join([o["label"] for o in structural_objects])
        exclusion_text = self._build_exclusion_zones(structural_objects)

        # Auto-generate reinforcement from technical_spec
        reinforcement = self._build_reinforcement(style_key, spec, window_wall, door_wall)

        # Count furniture types for anti-duplication
        from collections import Counter
        label_counts = Counter(furniture_labels)
        count_str = ", ".join([f"{count} {label}{'s' if count > 1 else ''}" for label, count in label_counts.items()])

        prompt = f"""TASK: Edit this 2D top-down floor plan to show the "{spec['name']}" arrangement.

╔══════════════════════════════════════════════════════════════╗
║  ⚠️  STRICT RULES                                            ║
╠══════════════════════════════════════════════════════════════╣
║  1. Exactly {movable_count} movable items must appear.           ║
║  2. REQUIRED: [{furniture_list_str}]                             ║
║  3. DO NOT ADD or REMOVE any furniture.                      ║
║  4. DOOR ({door_wall} {door_pct}): 2+ feet clear.               ║
║  5. DO NOT MOVE: [{structural_list}]                             ║
║  6. OUTPUT = 2D TOP-DOWN VIEW (not 3D).                      ║
╚══════════════════════════════════════════════════════════════╝

⚠️ DO NOT DUPLICATE FURNITURE:
- This room has exactly: {count_str}.
- MOVE = ERASE from old spot + PLACE in new spot. Do NOT copy.
- If the same item appears in TWO places, the output is INVALID.
- COUNT each furniture type before finishing. Counts must match input.

{exclusion_text}

STYLE: {spec['name']}
{layout_plan.get('description', spec['description'])}

FURNITURE PLACEMENT — FOLLOW EXACTLY:
{placement_text}

ZONES:
{zone_text}

{reinforcement}

DOOR CLEARANCE: {layout_plan.get('door_clearance', 'Keep door clear')}

STRUCTURAL (fixed): Door={door_wall} {door_pct}, Window={window_wall} {window_pct}, Others=[{structural_list}]

OUTPUT: Same 2D style as input, all {movable_count} items in their NEW positions, professional floor plan.
Do NOT place furniture on top of kitchen appliances, bathroom fixtures, or any structural element.
Edit the floor plan to show "{spec['name']}"."""

        _save_debug_json(f"{self._debug_ts}_image_{style_key}_INPUT.json", {
            "style_key": style_key, "reinforcement": reinforcement,
            "exclusion_zones": exclusion_text, "full_prompt": prompt,
        })

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        try:
            image_data = base64.b64decode(image_base64)
            response = await asyncio.to_thread(
                self.client.models.generate_content, model=self.image_model,
                contents=[types.Part.from_bytes(data=image_data, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"], temperature=0.3)
            )
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return base64.b64encode(part.inline_data.data).decode('utf-8')
            return None
        except Exception as e:
            print(f"[Designer] Image error {style_key}: {e}")
            import traceback; traceback.print_exc()
            return None


# ============================================================================
# LANGGRAPH NODES
# ============================================================================
@traceable(name="designer_node", run_type="chain", tags=["langgraph", "node"])
async def designer_node(state: AgentState) -> Dict[str, Any]:
    designer = InteriorDesignerAgent()
    try:
        variations = await designer.generate_layout_variations(
            current_layout=state["current_layout"], room_dims=state["room_dimensions"],
            locked_ids=state.get("locked_object_ids", []), image_base64=state.get("image_base64")
        )
        return {
            "layout_variations": variations,
            "proposed_layout": variations[0]["layout"] if variations else state["current_layout"],
            "explanation": f"Generated {len(variations)} layout variations.",
            "should_continue": False,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"Designer failed: {str(e)}", "should_continue": False}

def designer_node_sync(state: AgentState) -> Dict[str, Any]:
    return asyncio.run(designer_node(state))