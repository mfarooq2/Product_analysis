import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional
from pydantic import BaseModel, Field
from urllib.parse import quote_plus, urljoin
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("product_search.log", mode="a")],
)
logger = logging.getLogger(__name__)


# Pydantic Model for structured product data
class Product(BaseModel):
    title: str
    price: Optional[float]
    url: str


class Platform:
    """Base class for e-commerce platforms"""

    def __init__(self, name: str, base_url: str, search_path: str):
        self.name = name
        self.base_url = base_url
        self.search_path = search_path
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def get_search_url(self, query: str) -> str:
        """Generates search URL for platform"""
        return urljoin(self.base_url, self.search_path + quote_plus(query))

    def extract_product_info(self, item) -> Product:
        """Helper function to extract common product info"""
        title_elem = item.select_one("h3.s-item__title")
        title = title_elem.text.strip() if title_elem else "No Title"
        url = item.select_one("a.s-item__link")["href"]
        price_elem = item.select_one("span.s-item__price")
        price = (
            float(re.sub(r"[^\d.]", "", price_elem.text.strip()))
            if price_elem
            else None
        )
        return Product(title=title, price=price, url=url)

    async def make_request(
        self, url: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """Async HTTP request"""
        try:
            async with session.get(url, headers=self.headers, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.warning(f"Error making request to {url}: {str(e)}")
        return None

    def parse_search_results(self, html: str) -> List[Product]:
        """Parse the HTML response and return product details"""
        pass


class Amazon(Platform):
    def __init__(self):
        super().__init__(
            name="Amazon", base_url="https://www.amazon.com", search_path="/s?k="
        )

    def parse_search_results(self, html: str) -> List[Product]:
        """Parse Amazon search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select('div[data-component-type="s-search-result"]'):
            try:
                product = self.extract_product_info(item)
                results.append(product)
            except Exception as e:
                logger.error(f"Error parsing Amazon result: {str(e)}", exc_info=True)
        return results


class Ebay(Platform):
    def __init__(self):
        super().__init__(
            name="Ebay",
            base_url="https://www.ebay.com",
            search_path="/sch/i.html?_nkw=",
        )

    def parse_search_results(self, html: str) -> List[Product]:
        """Parse eBay search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("li.s-item"):
            try:
                product = self.extract_product_info(item)
                results.append(product)
            except Exception as e:
                logger.error(f"Error parsing eBay result: {str(e)}", exc_info=True)
        return results


class BestBuy(Platform):
    def __init__(self):
        super().__init__(
            name="BestBuy",
            base_url="https://www.bestbuy.com",
            search_path="/site/searchpage.jsp?st=",
        )

    def parse_search_results(self, html: str) -> List[Product]:
        """Parse Best Buy search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("li.sku-item"):
            try:
                product = self.extract_product_info(item)
                results.append(product)
            except Exception as e:
                logger.error(f"Error parsing Best Buy result: {str(e)}", exc_info=True)
        return results


class ProductSearchTool:
    """Tool for searching products across platforms"""

    def __init__(self):
        self.platforms = [Amazon(), Ebay(), BestBuy()]

    async def _search_platform(
        self, platform: Platform, query: str, session: aiohttp.ClientSession
    ) -> List[Product]:
        """Search a single platform asynchronously"""
        try:
            url = platform.get_search_url(query)
            html = await platform.make_request(url, session)
            if html:
                return platform.parse_search_results(html)
        except Exception as e:
            logger.error(f"Error searching {platform.name}: {str(e)}", exc_info=True)
        return []

    async def search_products(self, query: str) -> str:
        """Search products across all platforms asynchronously"""
        all_results = []
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._search_platform(platform, query, session)
                for platform in self.platforms
            ]
            results = await asyncio.gather(*tasks)
            for platform_results in results:
                all_results.extend(platform_results)

        if not all_results:
            logger.info("No results found.")
        return json.dumps([product.dict() for product in all_results])


class ProductComparisonTool:
    """Tool for comparing and analyzing product search results"""

    def __init__(self, model_identifier: str = "gpt-3.5-turbo"):
        self.model_identifier = model_identifier

    def compare_products(self, search_results: str) -> str:
        """Use model for analysis"""
        if not search_results:
            raise ValueError("search_results cannot be empty")
        try:
            prompt = f"Analyze the following product listings: {search_results}"
            return f"Comparison result: {prompt}"
        except Exception as e:
            logger.error(f"Error comparing products: {str(e)}", exc_info=True)
            return f"Error comparing products: {str(e)}"


class ProductSearchAgent:
    def __init__(self, model_name: str):
        self.search_tool = ProductSearchTool()
        self.comparison_tool = ProductComparisonTool()
        self.model_name = model_name

    async def search_and_compare(self, query: str) -> str:
        search_results = await self.search_tool.search_products(query)
        return self.comparison_tool.compare_products(search_results)


# Main entry point
async def main():
    # Initialize the Product Search Agent
    agent = ProductSearchAgent(model_name="google/flan-t5-xxl")
    query = input("Enter your query: ")
    # Perform a search query
    result = await agent.search_and_compare(query)
    # Print the result
    print(f"Comparison result for '{query}':\n{result}")


if __name__ == "__main__":
    # Run the main function using asyncio
    asyncio.run(main())
