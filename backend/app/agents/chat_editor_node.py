"""
Chat Editor Node

Conversational image editing agent that allows natural language
commands to modify the rendered room perspective.

FULLY TRACED with LangSmith - including command parsing and image edits.

EDIT TYPES:
- layout: move, rotate, reposition furniture (modifies bbox data)
- cosmetic: change colors, lighting, style (edits image directly)
- replace: swap one furniture for another at the EXACT same position
- remove: delete a piece of furniture from layout and image
"""

import json
import base64
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from google import genai
from google.genai import types

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


class ChatEditor:
    """
    Conversational editing agent for room layouts and renders.
    All methods are traced with LangSmith.
    """
    
    def __init__(self):
        from app.config import get_settings
        from app.tools.edit_image import EditImageTool
        
        settings = get_settings()
        api_key = settings.google_api_key
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        self.client = genai.Client(api_key=api_key)
        self.reasoning_model = settings.planning_model_name
        self.render_image_model_name = settings.render_image_model_name
        self.edit_tool = EditImageTool()

    @traceable(
        name="chat_editor.process_edit_command", 
        run_type="chain", 
        tags=["chat", "edit", "command"],
        metadata={"description": "Process natural language edit command"}
    )
    async def process_edit_command(
        self,
        command: str,
        current_layout: List[RoomObject],
        room_dims: RoomDimensions,
        current_image_base64: Optional[str] = None,
        layout_plan: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language editing command.
        TRACED: Full chain with command parsing and edit application.
        """
        # First, classify the edit type (traced)
        edit_type, parsed_command = await self._parse_command(command, current_layout, layout_plan)
        
        if edit_type == "layout":
            # Structural edit - modify layout positions
            updated_layout, explanation = await self._apply_layout_edit(
                parsed_command, current_layout, room_dims
            )
            return {
                "edit_type": "layout",
                "updated_layout": updated_layout,
                "updated_image_base64": None,
                "explanation": explanation,
                "needs_rerender": True
            }
        elif edit_type == "remove":
            # Remove object from layout AND erase from image
            updated_layout, removed_label, explanation = await self._apply_remove_edit(
                parsed_command, current_layout
            )
            # Also edit the image to visually erase the object
            updated_image = None
            if current_image_base64 and removed_label:
                updated_image, img_explanation = await self._apply_remove_from_image(
                    removed_label, current_image_base64
                )
                explanation += f" {img_explanation}"
            return {
                "edit_type": "remove",
                "updated_layout": updated_layout,
                "updated_image_base64": updated_image,
                "explanation": explanation,
                "needs_rerender": updated_image is None and removed_label is not None
            }
        elif edit_type == "replace":
            # Furniture replacement - remove old, add new at same position
            if current_image_base64:
                updated_image, explanation = await self._apply_replace_edit(
                    parsed_command, current_image_base64, layout_plan
                )
                return {
                    "edit_type": "cosmetic",
                    "updated_layout": current_layout,
                    "updated_image_base64": updated_image,
                    "explanation": explanation,
                    "needs_rerender": False
                }
            else:
                return {
                    "edit_type": "error",
                    "updated_layout": current_layout,
                    "updated_image_base64": None,
                    "explanation": "No rendered image available. Please generate a perspective view first.",
                    "needs_rerender": True
                }
        else:
            # Cosmetic edit - modify the image directly
            if current_image_base64:
                updated_image, explanation = await self._apply_image_edit(
                    parsed_command, current_image_base64
                )
                return {
                    "edit_type": "cosmetic",
                    "updated_layout": current_layout,
                    "updated_image_base64": updated_image,
                    "explanation": explanation,
                    "needs_rerender": False
                }
            else:
                return {
                    "edit_type": "error",
                    "updated_layout": current_layout,
                    "updated_image_base64": None,
                    "explanation": "No rendered image available for cosmetic editing. Please generate a render first.",
                    "needs_rerender": True
                }

    @traceable(
        name="gemini_parse_command", 
        run_type="llm", 
        tags=["gemini", "chat", "parsing", "api-call"],
        metadata={"model_type": "gemini-pro", "task": "command_parsing"}
    )
    async def _parse_command(
        self, 
        command: str, 
        current_layout: List[RoomObject],
        layout_plan: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse natural language command into structured edit instruction.
        TRACED as an LLM call.
        """
        furniture_list = [{"id": obj.id, "label": obj.label} for obj in current_layout]
        
        # Format design context from layout_plan if available
        design_context = ""
        if layout_plan:
            design_context = f"""
DESIGN CONTEXT (The current layout follows this plan):
- Concept: {layout_plan.get('concept_name', 'Custom Layout')}
- Description: {layout_plan.get('description', 'User generated layout')}
- Intent (Furniture Placement): {json.dumps(layout_plan.get('furniture_placement', {}), indent=2)}
"""

        prompt = f"""You are an interior design assistant parsing user edit commands.

CURRENT FURNITURE IN ROOM:
{json.dumps(furniture_list, indent=2)}
{design_context}

USER COMMAND: "{command}"

Classify this command and parse it into a structured format.

EDIT TYPES:
1. "layout" - Commands that move, rotate, or reposition furniture
   Examples: "move desk to left", "rotate bed 90 degrees", "swap desk and dresser positions"
   
2. "cosmetic" - Commands that change appearance without moving or replacing furniture
   Examples: "make it more cozy", "change lighting", "make rug blue", "add plants"

3. "replace" - Commands that replace/swap/change one piece of furniture INTO a different type of furniture
   Examples: "change the table into a workdesk", "replace the sofa with an armchair",
             "turn the chair into a bean bag", "swap the nightstand for a bookshelf",
             "make the desk a standing desk", "convert the table to a dining table"
   KEY: The user wants the OLD furniture REMOVED and a NEW different furniture placed at the EXACT SAME position.

4. "remove" - Commands that DELETE/REMOVE a piece of furniture entirely from the room
   Examples: "remove the desk", "delete the rug", "get rid of the nightstand",
             "take out the lamp", "I don't want the dresser", "remove chair_1",
             "clear the coffee table", "take away the plant"
   KEY: The user wants a specific object COMPLETELY GONE from the room — not replaced, not moved, just removed.
   IMPORTANT: For remove, you MUST set target_object_id to the matching furniture id from the list above.
   If the user says a label like "desk", find the matching id (e.g. "desk_1") from the furniture list.

Return JSON:
{{
  "edit_type": "layout" | "cosmetic" | "replace" | "remove",
  "action": "move" | "rotate" | "style" | "add" | "remove" | "replace",
  "target_object_id": "id of the furniture being acted on, or null",
  "parameters": {{
    "direction": "left|right|up|down" (for move),
    "distance": "small|medium|large" (for move),
    "rotation": 90 (degrees, for rotate),
    "style_change": "description" (for cosmetic),
    "old_furniture": "what the current furniture is" (for replace),
    "new_furniture": "what it should become" (for replace)
  }},
  "natural_description": "Human-readable description of the change"
}}"""

        try:
            response = self.client.models.generate_content(
                model=self.reasoning_model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            parsed = json.loads(response.text)
            return parsed.get("edit_type", "cosmetic"), parsed
            
        except Exception as e:
            # Default to cosmetic if parsing fails
            return "cosmetic", {
                "edit_type": "cosmetic",
                "action": "style",
                "natural_description": command
            }

    @traceable(name="apply_layout_edit", run_type="chain", tags=["edit", "layout"])
    async def _apply_layout_edit(
        self,
        parsed_command: Dict[str, Any],
        current_layout: List[RoomObject],
        room_dims: RoomDimensions
    ) -> Tuple[List[RoomObject], str]:
        """
        Apply a structural layout edit.
        TRACED: Layout modification logic.
        """
        target_id = parsed_command.get("target_object_id")
        action = parsed_command.get("action", "move")
        params = parsed_command.get("parameters", {})
        
        # Find target object
        target_obj = None
        if target_id:
            target_obj = next((o for o in current_layout if o.id == target_id), None)
        
        if not target_obj and action in ["move", "rotate"]:
            return current_layout, f"Could not find target object for edit. Available: {[o.label for o in current_layout]}"
        
        # Create updated layout
        updated_layout = []
        explanation = ""
        
        for obj in current_layout:
            new_obj = RoomObject(
                id=obj.id,
                label=obj.label,
                bbox=obj.bbox.copy(),
                type=obj.type,
                orientation=obj.orientation,
                is_locked=obj.is_locked,
                z_index=obj.z_index,
                material_hint=obj.material_hint
            )
            
            if target_obj and obj.id == target_obj.id:
                if action == "move":
                    direction = params.get("direction", "")
                    distance_map = {"small": 5, "medium": 10, "large": 20}
                    distance = distance_map.get(params.get("distance", "medium"), 10)
                    
                    if direction == "left":
                        new_obj.bbox[0] = max(0, new_obj.bbox[0] - distance)
                    elif direction == "right":
                        new_obj.bbox[0] = min(100 - new_obj.bbox[2], new_obj.bbox[0] + distance)
                    elif direction == "up":
                        new_obj.bbox[1] = max(0, new_obj.bbox[1] - distance)
                    elif direction == "down":
                        new_obj.bbox[1] = min(100 - new_obj.bbox[3], new_obj.bbox[1] + distance)
                    
                    explanation = f"Moved {obj.label} {direction} by {distance}%"
                
                elif action == "rotate":
                    rotation = params.get("rotation", 90)
                    new_obj.orientation = (new_obj.orientation + rotation) % 360
                    explanation = f"Rotated {obj.label} by {rotation} degrees (now facing {new_obj.orientation}deg)"
            
            updated_layout.append(new_obj)
        
        if not explanation:
            explanation = f"Processed command: {parsed_command.get('natural_description', 'Unknown edit')}"
        
        return updated_layout, explanation

    # ========================================================================
    # REMOVE EDIT — remove object from layout data
    # ========================================================================
    @traceable(name="apply_remove_edit", run_type="chain", tags=["edit", "remove"])
    async def _apply_remove_edit(
        self,
        parsed_command: Dict[str, Any],
        current_layout: List[RoomObject],
    ) -> Tuple[List[RoomObject], Optional[str], str]:
        """
        Remove a furniture item from the layout.

        Returns:
            (updated_layout, removed_object_label_or_None, explanation)
        """
        target_id = parsed_command.get("target_object_id")

        # Try to find the target object
        target_obj = None
        if target_id:
            target_obj = next((o for o in current_layout if o.id == target_id), None)

        # Fallback: try matching by label from natural_description
        if not target_obj:
            desc = parsed_command.get("natural_description", "").lower()
            for obj in current_layout:
                if obj.label.lower() in desc or obj.id.lower() in desc:
                    target_obj = obj
                    break

        if not target_obj:
            available = [f"{o.id} ({o.label})" for o in current_layout if o.type.value == "movable"]
            return (
                current_layout,
                None,
                f"Could not find the object to remove. Available movable objects: {', '.join(available)}"
            )

        # Prevent removing structural objects
        if target_obj.type.value == "structural":
            return (
                current_layout,
                None,
                f"Cannot remove {target_obj.label} ({target_obj.id}) — it is a structural element "
                f"(door, window, wall, fixture). Only movable furniture can be removed."
            )

        # Filter out the target object
        updated_layout = [o for o in current_layout if o.id != target_obj.id]
        removed_label = target_obj.label
        explanation = f"Removed {removed_label} ({target_obj.id}) from the room."

        return updated_layout, removed_label, explanation

    # ========================================================================
    # REMOVE FROM IMAGE — erase the object visually using Gemini
    # ========================================================================
    @traceable(
        name="apply_remove_from_image",
        run_type="llm",
        tags=["gemini", "image", "edit", "remove", "api-call"],
        metadata={"model_type": "gemini-image", "task": "object_removal"}
    )
    async def _apply_remove_from_image(
        self,
        removed_label: str,
        current_image_base64: str
    ) -> Tuple[str, str]:
        """
        Remove an object from the rendered image using Gemini image editing.
        The area where the object was is filled with appropriate floor/background.

        TRACED as an LLM/image-gen call.
        """
        prompt = f"""Edit this interior room photograph. Remove a piece of furniture.

TASK: Completely remove the {removed_label} from this room image.

CRITICAL RULES:
1. Find the {removed_label} in the image and ERASE it entirely.
2. Fill the area where the {removed_label} was with the appropriate background:
   - If it was on the floor, show the floor surface (wood, tile, carpet, etc.)
   - If it was against a wall, show the wall behind it.
   - Blend seamlessly with the surrounding area — no artifacts, no outlines.
3. Do NOT move, resize, or alter any OTHER furniture or objects.
4. Maintain the same camera angle, lighting, shadows, and perspective.
5. Maintain photorealistic quality throughout.
6. The result should look as if the {removed_label} was never there.

Generate the edited room photograph with the {removed_label} removed."""

        try:
            if "," in current_image_base64:
                current_image_base64 = current_image_base64.split(",")[1]

            image_data = base64.b64decode(current_image_base64)

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.render_image_model_name,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                    temperature=0.2,
                )
            )

            if (response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        new_image = base64.b64encode(part.inline_data.data).decode('utf-8')
                        return new_image, f"Visually erased the {removed_label} from the image."

            return current_image_base64, f"Could not erase {removed_label} from image — model returned no image."

        except Exception as e:
            return current_image_base64, f"Image removal failed: {str(e)}"

    @traceable(
        name="apply_replace_edit",
        run_type="llm",
        tags=["gemini", "image", "edit", "replace", "api-call"],
        metadata={"model_type": "gemini-image", "task": "furniture_replacement"}
    )
    async def _apply_replace_edit(
        self,
        parsed_command: Dict[str, Any],
        current_image_base64: str,
        layout_plan: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Replace one piece of furniture with another at the EXACT same position.
        The image model is instructed to:
        1. Identify the old furniture in the image
        2. Remove it completely
        3. Place the new furniture at the exact same location, same size footprint
        4. Keep everything else untouched
        
        TRACED as an LLM/image-gen call.
        """
        params = parsed_command.get("parameters", {})
        old_furniture = params.get("old_furniture", "furniture")
        new_furniture = params.get("new_furniture", "furniture")
        target_id = parsed_command.get("target_object_id", "")
        description = parsed_command.get("natural_description", f"Replace {old_furniture} with {new_furniture}")

        location_context = ""
        if layout_plan and "furniture_placement" in layout_plan and target_id:
             placement = layout_plan["furniture_placement"].get(target_id)
             if placement:
                 location_context = f"LOCATION CONTEXT: The {old_furniture} is currently located {placement}."

        prompt = f"""Edit this interior room photograph. Replace one piece of furniture with another.

TASK: Replace the {old_furniture} with a {new_furniture}.
{location_context}

╔══════════════════════════════════════════════════════════════╗
  ⛔ CRITICAL POSITION RULES — MUST FOLLOW EXACTLY
╚══════════════════════════════════════════════════════════════╝
  1. Find the {old_furniture} in the image.
  2. COMPLETELY REMOVE the {old_furniture} from that spot.
  3. Place a {new_furniture} at the EXACT SAME POSITION as the {old_furniture}.
  4. Do NOT move the new furniture to a different spot.
  5. Do NOT change anything else in the room.
╔══════════════════════════════════════════════════════════════╗

REQUIREMENTS:
- The {new_furniture} must look realistic and match the room's style.
- Maintain photorealistic quality — same lighting, shadows, perspective.
- Keep the same camera angle exactly.
- Preserve all other furniture, walls, floor, and decorations.
- The ONLY change should be: {old_furniture} → {new_furniture} at the same spot.

Generate the edited room photograph."""

        try:
            if "," in current_image_base64:
                current_image_base64 = current_image_base64.split(",")[1]

            image_data = base64.b64decode(current_image_base64)

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.render_image_model_name,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                    temperature=0.2,
                )
            )

            if (response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        new_image = base64.b64encode(part.inline_data.data).decode('utf-8')
                        return new_image, f"Replaced {old_furniture} with {new_furniture} at the same position."

            return current_image_base64, f"Could not generate replacement image. The model returned no image."

        except Exception as e:
            return current_image_base64, f"Replace edit failed: {str(e)}"

    @traceable(
        name="gemini_image_edit", 
        run_type="llm", 
        tags=["gemini", "image", "edit", "api-call"],
        metadata={"model_type": "gemini-image", "task": "cosmetic_edit"}
    )
    async def _apply_image_edit(
        self,
        parsed_command: Dict[str, Any],
        current_image_base64: str
    ) -> Tuple[str, str]:
        """
        Apply a cosmetic edit to the rendered image.
        TRACED as an LLM/image-gen call.
        """
        edit_description = parsed_command.get("natural_description", "Apply the requested change")
        
        try:
            new_image = await self.edit_tool.edit_image(
                base_image=current_image_base64,
                instruction=edit_description
            )
            return new_image, f"Applied: {edit_description}"
            
        except Exception as e:
            return current_image_base64, f"Edit failed: {str(e)}"


# LangGraph node functions

@traceable(name="chat_editor_node", run_type="chain", tags=["langgraph", "node", "chat"])
async def chat_editor_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node for conversational editing.
    TRACED: Full trace with command processing.
    """
    editor = ChatEditor()
    
    edit_command = state.get("edit_command", "")
    if not edit_command:
        return {
            "explanation": "No edit command provided. Use natural language to describe changes."
        }
    
    try:
        result = await editor.process_edit_command(
            command=edit_command,
            current_layout=state.get("current_layout", []),
            room_dims=state["room_dimensions"],
            current_image_base64=state.get("output_image_base64")
        )
        
        updates = {
            "explanation": result["explanation"],
            "should_continue": result.get("needs_rerender", False)
        }
        
        if result["edit_type"] in ("layout", "remove") and result["updated_layout"]:
            updates["current_layout"] = result["updated_layout"]
            updates["proposed_layout"] = result["updated_layout"]
        
        if result["updated_image_base64"]:
            updates["output_image_base64"] = result["updated_image_base64"]
        
        return updates
        
    except Exception as e:
        return {
            "error": f"Chat editor failed: {str(e)}",
            "explanation": f"Could not process edit command: {str(e)}"
        }


def chat_editor_node_sync(state: AgentState) -> Dict[str, Any]:
    """Synchronous wrapper for LangGraph compatibility."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(chat_editor_node(state))
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, chat_editor_node(state))
            return future.result()