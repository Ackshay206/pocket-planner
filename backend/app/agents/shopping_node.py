"""
Shopping Agent Node

Agentic node that:
1. Analyzes the perspective image to describe each furniture item's style
2. Allocates the total room budget across items by importance/size
3. Searches Google Shopping (via SerpAPI) for each item
4. Returns structured product recommendations

FULLY TRACED with LangSmith.
"""

import json
import base64
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from app.config import get_settings
from app.models.room import RoomObject
from app.tools.serp_search import SerpSearchTool

try:
    from langsmith import traceable
    LANGSMITH_ENABLED = True
except ImportError:
    LANGSMITH_ENABLED = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class ShoppingAgent:
    """
    AI agent that finds real products matching the furniture in a room render.
    """

    def __init__(self):
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = settings.planning_model_name
        self.search_tool = SerpSearchTool()
        print(f"[ShoppingAgent] Initialized with model: {self.model}")

    @traceable(
        name="shopping_agent.find_products",
        run_type="chain",
        tags=["shopping", "agent", "pipeline"],
    )
    async def find_products(
        self,
        current_layout: List[RoomObject],
        total_budget: float,
        perspective_image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Analyzes room, allocates budget, searches products.
        """
        # Step 1: Get only movable furniture
        movable_items = [
            {"id": obj.id, "label": obj.label}
            for obj in current_layout
            if obj.type.value == "movable"
        ]

        print(f"[ShoppingAgent] Movable items: {movable_items}")
        print(f"[ShoppingAgent] Total budget: ${total_budget}")
        print(f"[ShoppingAgent] Has perspective image: {perspective_image_base64 is not None}")

        if not movable_items:
            return {"items": [], "total_estimated": 0, "message": "No movable furniture found."}

        # Step 2: Ask Gemini to describe items + allocate budget
        item_descriptions = await self._describe_and_allocate(
            movable_items, total_budget, perspective_image_base64
        )

        print(f"[ShoppingAgent] Gemini returned {len(item_descriptions)} item descriptions:")
        for desc in item_descriptions:
            print(f"  - {desc.get('id')}: query=\"{desc.get('search_query')}\" budget=${desc.get('budget')}")

        # Step 3: Search for each item in parallel
        search_tasks = [self._search_for_item(item) for item in item_descriptions]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Step 4: Assemble results
        items = []
        total_estimated = 0.0
        for i, result in enumerate(search_results):
            desc = item_descriptions[i]
            if isinstance(result, Exception):
                print(f"[ShoppingAgent] Search FAILED for {desc['id']}: {result}")
                items.append({
                    "furniture_id": desc["id"],
                    "furniture_label": desc["label"],
                    "search_query": desc.get("search_query", ""),
                    "budget_allocated": desc.get("budget", 0),
                    "products": [],
                    "error": str(result),
                })
            else:
                best_price = result[0]["price"] if result and result[0].get("price") else 0
                total_estimated += best_price
                items.append({
                    "furniture_id": desc["id"],
                    "furniture_label": desc["label"],
                    "search_query": desc.get("search_query", ""),
                    "budget_allocated": desc.get("budget", 0),
                    "products": result,
                })

        return {
            "items": items,
            "total_estimated": round(total_estimated, 2),
            "total_budget": total_budget,
            "message": f"Found products for {len([i for i in items if i['products']])} of {len(movable_items)} items.",
        }

    @traceable(
        name="gemini_describe_and_allocate",
        run_type="llm",
        tags=["gemini", "shopping", "description", "budget"],
    )
    async def _describe_and_allocate(
        self,
        movable_items: List[Dict[str, str]],
        total_budget: float,
        image_base64: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use Gemini to convert generic labels to specific product queries
        and allocate budget proportionally.

        No fallback — raises on failure so we can debug properly.
        """
        item_list_str = json.dumps(movable_items, indent=2)
        num_items = len(movable_items)

        prompt = f"""You are a furniture shopping assistant. Convert generic furniture labels into specific Google Shopping search queries and allocate a budget.

TOTAL BUDGET: ${total_budget:.2f}

FURNITURE ITEMS (from plan):
{item_list_str}

TASK 1 — Analyze the attached perspective image:
Detect ANY additional furniture or decor items visible in the image that are NOT in the "FURNITURE ITEMS" list above (e.g., rugs, plants, lamps, artwork, pillows).
For each new item found, create a new entry with a unique ID (e.g., "detected_plant_1").

TASK 2 — Write a Google Shopping search query for each item (original + detected):
Generate a Google Shopping search query. The specificity depends on the item category, but MUST always be specific enough to find a real product (never generic).

1. KEY FURNITURE (Bed, Sofa, Desk, Wardrobe, Dining Table):
   - HIGH SPECIFICITY. Include precise style, material, color, size, and defining features.
   - Example: "Bed" → "Queen size walnut platform bed frame" : maximum 6 words

2. SECONDARY FURNITURE (Nightstand, Chair, Coffee Table, Dresser):
   - MODERATE SPECIFICITY. Include style, material, color, and main dimension.
   - Example: "Nightstand" → "White oak bedside table with drawers" : maximum 5 words

3. DECOR / ACCESSORIES (Rug, Lamp, Plant, Artwork):
   - LOW SPECIFICITY. Focus on style, color, type, and size.
   - Example: "Plant" → "Artificial fiddle leaf fig tree" :maximum 4 words

Examples of POOR/GENERIC queries (AVOID THESE): "bed", "blue sofa", "wooden table", "plant".

TASK 3 — Allocate ${total_budget:.2f} across ALL items (original + detected) proportionally:
- Prioritize key furniture (Bed, Sofa, Desk) (~70% of budget divided equally)
- Secondary furniture (Tables, Chairs, Dressers) (~60% of budget )
- Decor/Accessories (Plants, Rugs, Lamps) (~30% of budget)

Total estimated cost MUST sum to exactly plus or minus 10% of ${total_budget:.2f}.

Return ONLY a JSON array with ALL objects (original items + detected items):
[
  {{"id": "bed_1", "label": "bed", "search_query": "queen size walnut platform bed frame", "budget": 500.00}},
  {{"id": "detected_plant_1", "label": "plant", "search_query": "artificial fiddle leaf fig tree 6ft", "budget": 80.00, "is_new": true}}
]

RULES:
- Every original item from the input list MUST appear exactly once.
- "id" and "label" of original items must match EXACTLY.
- For new detected items, use IDs starting with "detected_".
- Total estimated cost MUST sum to exactly plus or minus 10% of ${total_budget:.2f}.
- Return ONLY the JSON array, nothing else."""

        # Build contents
        contents = []
        if image_base64:
            clean_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64 
            try:
                img_data = base64.b64decode(clean_b64)
                contents.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))
                print(f"[ShoppingAgent] Attached perspective image ({len(img_data)} bytes)")
            except Exception as e:
                print(f"[ShoppingAgent] WARNING: Failed to decode image: {e}")
        contents.append(prompt)

        print(f"[ShoppingAgent] Calling Gemini model={self.model} with {len(contents)} content parts")
        print(f"[ShoppingAgent] Prompt length: {len(prompt)} chars")

        # Call Gemini
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                )
            )
        except Exception as e:
            print(f"[ShoppingAgent] ERROR: Gemini API call failed!")
            print(f"[ShoppingAgent] Exception type: {type(e).__name__}")
            print(f"[ShoppingAgent] Exception: {e}")
            print(f"[ShoppingAgent] Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Gemini API call failed: {e}") from e

        # Parse response
        raw_text = response.text
        print(f"[ShoppingAgent] Gemini raw response ({len(raw_text)} chars): {raw_text[:1000]}")

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"[ShoppingAgent] ERROR: Failed to parse Gemini response as JSON!")
            print(f"[ShoppingAgent] JSONDecodeError: {e}")
            print(f"[ShoppingAgent] Full raw response:\n{raw_text}")
            raise RuntimeError(f"Gemini returned invalid JSON: {e}\nRaw: {raw_text[:500]}") from e

        # Handle Gemini wrapping the array in a dict
        if isinstance(result, dict):
            print(f"[ShoppingAgent] WARNING: Gemini returned dict with keys: {list(result.keys())}")
            # Try to extract the array from common wrapper keys
            extracted = None
            for key in ("items", "furniture", "results", "data", "furniture_items"):
                if key in result and isinstance(result[key], list):
                    extracted = result[key]
                    print(f"[ShoppingAgent] Extracted array from key '{key}' ({len(extracted)} items)")
                    break
            if extracted is None:
                # If dict has the expected item fields, it might be a single item — wrap it
                if "id" in result and "search_query" in result:
                    extracted = [result]
                    print(f"[ShoppingAgent] Wrapped single dict item into list")
                else:
                    raise RuntimeError(f"Gemini returned dict but no extractable array. Keys: {list(result.keys())}. Full response: {raw_text[:500]}")
            result = extracted

        if not isinstance(result, list):
            raise RuntimeError(f"Expected list from Gemini, got {type(result).__name__}: {raw_text[:500]}")

        if len(result) == 0:
            raise RuntimeError(f"Gemini returned empty list. Raw: {raw_text[:500]}")

        print(f"[ShoppingAgent] Parsed {len(result)} items from Gemini")

        # Validate each item has required fields
        for i, item in enumerate(result):
            if not isinstance(item, dict):
                raise RuntimeError(f"Item {i} is not a dict: {item}")
            missing = [k for k in ("id", "label", "search_query", "budget") if k not in item]
            if missing:
                print(f"[ShoppingAgent] WARNING: Item {i} missing fields {missing}: {item}")
                # Try to fill in missing fields from movable_items
                if "id" not in item and i < len(movable_items):
                    item["id"] = movable_items[i]["id"]
                if "label" not in item and i < len(movable_items):
                    item["label"] = movable_items[i]["label"]
                if "search_query" not in item:
                    item["search_query"] = f"{item.get('label', 'furniture')} for home"
                if "budget" not in item:
                    item["budget"] = round(total_budget / num_items, 2)

        # Validate + fix budget sum
        budget_sum = sum(item.get("budget", 0) for item in result)
        print(f"[ShoppingAgent] Budget sum: ${budget_sum:.2f} (expected: ${total_budget:.2f})")

        if abs(budget_sum - total_budget) > 1.0:
            print(f"[ShoppingAgent] WARNING: Budget sum off by ${abs(budget_sum - total_budget):.2f}, rescaling")
            if budget_sum > 0:
                ratio = total_budget / budget_sum
                for item in result:
                    item["budget"] = round(item.get("budget", 0) * ratio, 2)
            # Fix rounding on last item
            new_sum = sum(item["budget"] for item in result)
            diff = round(total_budget - new_sum, 2)
            if result and diff != 0:
                result[0]["budget"] = round(result[0]["budget"] + diff, 2)

        return result

    @traceable(
        name="search_for_item",
        run_type="tool",
        tags=["serpapi", "shopping", "search"],
    )
    async def _search_for_item(
        self,
        item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Search Google Shopping for a single item within its allocated budget.
        Retries with a broader query if first attempt returns no results.
        """
        query = item.get("search_query", "")
        budget = item.get("budget", 500)
        label = item.get("label", "furniture")

        if not query:
            print(f"[ShoppingAgent] WARNING: Empty search query for {item.get('id')}, using label")
            query = f"{label} furniture"

        print(f"[ShoppingAgent] Searching: \"{query}\" (budget: ${budget})")

        # First attempt with Gemini's specific query
        products = await self.search_tool.search_shopping(
            query=query,
            max_price=budget,
            num_results=3,
        )

        # Agentic retry: if no results, broaden the query
        if not products:
            broader_query = f"{label}"
            print(f"[ShoppingAgent] No results for \"{query}\", retrying with \"{broader_query}\" (budget +50%)")
            products = await self.search_tool.search_shopping(
                query=broader_query,
                max_price=budget * 1.5,
                num_results=3,
            )

        print(f"[ShoppingAgent] Found {len(products)} products for {item.get('id')}")
        return products