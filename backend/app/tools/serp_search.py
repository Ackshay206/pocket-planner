"""
SerpAPI Google Shopping Search Tool

Searches Google Shopping for furniture products matching a query and budget.
Used by the Shopping Agent to find real products for the user's room.
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional

from app.config import get_settings

try:
    from langsmith import traceable
    LANGSMITH_ENABLED = True
except ImportError:
    LANGSMITH_ENABLED = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


SERPAPI_BASE_URL = "https://serpapi.com/search.json"


class SerpSearchTool:
    """
    Tool for searching Google Shopping via SerpAPI.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.serpapi_key
        if not self.api_key:
            raise ValueError("SERPAPI_KEY not set in .env file")

    @traceable(
        name="serp_search_tool.search_shopping",
        run_type="tool",
        tags=["tool", "serpapi", "shopping"],
        metadata={"description": "Search Google Shopping for a product"}
    )
    async def search_shopping(
        self,
        query: str,
        max_price: Optional[float] = None,
        num_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search Google Shopping for products.

        Args:
            query: Search query (e.g. "mid-century modern walnut nightstand")
            max_price: Maximum price in USD. Results above this are filtered out.
            num_results: Max results to return.

        Returns:
            List of product dicts with title, price, link, thumbnail, source, rating.
        """
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": self.api_key,
            "num": min(num_results * 3, 30),  # fetch extra to allow price filtering
            "hl": "en",
            "gl": "us",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(SERPAPI_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            shopping_results = data.get("shopping_results", [])
            
            # DEBUG: Inspect first item fields to find direct links
            if shopping_results:
                print(f"[SerpAPI Debug] First item keys: {list(shopping_results[0].keys())}")
                print(f"[SerpAPI Debug] First item link: {shopping_results[0].get('link')}")
                # Check for other potential link fields people might use
                for k in ['product_link', 'offer_link', 'merchant_link', 'display_link', 'dest_url']:
                    if k in shopping_results[0]:
                        print(f"[SerpAPI Debug]   {k}: {shopping_results[0][k]}")

            products = []
            for item in shopping_results:
                price_str = item.get("extracted_price")
                price = None
                if price_str is not None:
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError):
                        price = None

                # Filter by max_price if set
                if max_price and price is not None and price > max_price:
                    continue

                # Build the best available link:
                # 1. "link" — direct retailer URL (best, when available)
                # 2. "product_link" — Google Shopping product page (always works)
                # 3. Construct from product_id as last resort
                direct_link = item.get("link", "")
                product_link = item.get("product_link", "")
                product_id = item.get("product_id", "")

                best_link = ""
                if direct_link and not direct_link.startswith("https://www.google.com/aclk"):
                    # Direct retailer link (ideal)
                    best_link = direct_link
                elif product_link:
                    # Google Shopping product page
                    best_link = product_link
                elif product_id:
                    # Construct Google Shopping URL from product_id
                    best_link = f"https://www.google.com/shopping/product/{product_id}"
                elif direct_link:
                    # Google tracking link as last resort
                    best_link = direct_link

                products.append({
                    "title": item.get("title", "Unknown Product"),
                    "price": price,
                    "price_raw": item.get("price", ""),
                    "link": best_link,
                    "thumbnail": item.get("thumbnail", ""),
                    "source": item.get("source", ""),
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                })

                if len(products) >= num_results:
                    break

            return products

        except httpx.HTTPStatusError as e:
            print(f"[SerpAPI] HTTP error: {e.response.status_code} — {e.response.text[:200]}")
            return []
        except Exception as e:
            print(f"[SerpAPI] Search failed: {e}")
            return []