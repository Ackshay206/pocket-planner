"""
Render Node

Surgical image editing using Gemini 2.5 Flash Image.
Creates edited images by applying masked instructions to specific regions.
"""

import os
import json
import base64
from typing import List, Optional
from google import genai
from google.genai import types
import asyncio
from PIL import Image
import io
from pydantic import BaseModel, Field


class EditMask(BaseModel):
    """Single region edit instruction."""
    region_mask: str = Field(..., description="Base64 PNG with alpha channel marking edit region")
    instruction: str = Field(..., description="Edit instruction like 'Repaint wall navy blue'")


class ImageEditor:
    """
    Applies surgical edits to images using Gemini's image generation capabilities.
    
    Since direct mask-based inpainting may not be supported, we use a fallback
    approach that describes the region to edit based on the mask location.
    """
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash-image"  # Image generation model
    
    def _get_mask_region_description(self, mask_base64: str, image_width: int, image_height: int) -> str:
        """
        Analyze a mask to determine what region it covers.
        Returns a description like "top-left area", "center", "right wall", etc.
        """
        try:
            # Decode mask
            if "," in mask_base64:
                mask_base64 = mask_base64.split(",")[1]
            mask_data = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
            
            # Find the bounding box of the mask (non-white pixels = mask region)
            # Black pixels indicate the edit region
            mask_array = list(mask_image.getdata())
            width, height = mask_image.size
            
            min_x, min_y = width, height
            max_x, max_y = 0, 0
            
            for i, pixel in enumerate(mask_array):
                if pixel < 128:  # Dark pixel = mask region
                    x = i % width
                    y = i // width
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
            
            if min_x > max_x or min_y > max_y:
                return "the selected area"
            
            # Determine region description based on center of mask
            center_x = (min_x + max_x) / 2 / width
            center_y = (min_y + max_y) / 2 / height
            
            # Vertical position
            if center_y < 0.33:
                v_pos = "top"
            elif center_y > 0.66:
                v_pos = "bottom"
            else:
                v_pos = "center"
            
            # Horizontal position
            if center_x < 0.33:
                h_pos = "left"
            elif center_x > 0.66:
                h_pos = "right"
            else:
                h_pos = "center"
            
            if v_pos == "center" and h_pos == "center":
                return "the center area"
            elif v_pos == "center":
                return f"the {h_pos} side"
            elif h_pos == "center":
                return f"the {v_pos} area"
            else:
                return f"the {v_pos}-{h_pos} area"
                
        except Exception:
            return "the selected area"
    
    async def apply_edits(
        self, 
        base_image: str, 
        masks: List[EditMask],
        max_retries: int = 3
    ) -> str:
        """
        Apply surgical edits to an image based on masks and instructions.
        
        Args:
            base_image: Base64 encoded original image
            masks: List of EditMask objects with region_mask and instruction
            max_retries: Number of retry attempts
            
        Returns:
            Base64 encoded edited image
        """
        if not masks:
            return base_image
        
        # Handle data URL prefix
        if "," in base_image:
            base_image = base_image.split(",")[1]
        
        # Decode base image to get dimensions
        image_data = base64.b64decode(base_image)
        pil_image = Image.open(io.BytesIO(image_data))
        img_width, img_height = pil_image.size
        
        current_image = base_image
        
        # Apply each edit sequentially
        for mask in masks:
            # Get region description from mask
            region_desc = self._get_mask_region_description(
                mask.region_mask, img_width, img_height
            )
            
            # Create prompt for editing
            edit_prompt = f"""Edit this room image. 
Focus on {region_desc}: {mask.instruction}
Keep everything else exactly the same. 
Maintain photorealistic quality and consistent lighting."""
            
            # Decode current image
            if "," in current_image:
                current_image = current_image.split(",")[1]
            current_image_data = base64.b64decode(current_image)
            
            for attempt in range(max_retries):
                try:
                    # Generate edited image
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            types.Part.from_bytes(
                                data=current_image_data, 
                                mime_type="image/jpeg"
                            ),
                            edit_prompt
                        ],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"]
                        )
                    )
                    
                    # Extract image from response
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                current_image = base64.b64encode(
                                    part.inline_data.data
                                ).decode('utf-8')
                                break
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(f"Image editing failed: {e}")
        
        return current_image
    
    async def generate_layout_visualization(
        self,
        base_image: str,
        original_positions: dict,
        new_positions: dict
    ) -> str:
        """
        Generate a visualization showing furniture movement.
        
        This is a fallback when actual image editing isn't feasible.
        Could be enhanced to use overlays or arrows to show movement.
        """
        # For now, return the base image
        # Future enhancement: add visual overlays showing furniture movement
        return base_image
