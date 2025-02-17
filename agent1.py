import requests
import logging
from bs4 import BeautifulSoup
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from urllib.parse import quote_plus, urljoin
import re
import concurrent.futures
from transformers import pipeline
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("product_search.log")],
)
logger = logging.getLogger(__name__)


class Platform:
    """Base class for e-commerce platforms"""

    session = requests.Session()

    def __init__(self, name: str, base_url: str, search_path: str):
        self.name = name
        self.base_url = base_url
        self.search_path = search_path
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def make_request(self, url: str, retries: int = 3) -> Optional[str]:
        """Make HTTP request with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200 and response.content:
                    return response.text
                response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"All attempts failed for {url}")
                    return None
        return None

    def get_search_url(self, query: str) -> str:
        """Generates search URL for platform"""
        return urljoin(self.base_url, self.search_path + quote_plus(query))

    def parse_search_results(self, html: str) -> List[BaseModel]:
        """Parse the HTML response and return product details"""
        pass


class Amazon(Platform):
    def __init__(self):
        super().__init__(
            name="Amazon", base_url="https://www.amazon.com", search_path="/s?k="
        )

    def parse_search_results(self, html: str) -> List[BaseModel]:
        """Parse Amazon search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select('div[data-component-type="s-search-result"]'):
            try:
                title_elem = item.select_one("h2 a span")
                title = title_elem.text.strip() if title_elem else "No Title"
                url = urljoin(self.base_url, item.select_one("h2 a")["href"])
                price_elem = item.select_one("span.a-price span.a-offscreen")
                price = (
                    float(re.sub(r"[^\d.]", "", price_elem.text.strip()))
                    if price_elem
                    else None
                )
                results.append({"title": title, "price": price, "url": url})
            except Exception as e:
                logger.error(f"Error parsing Amazon result: {str(e)}")
        return results


class Ebay(Platform):
    def __init__(self):
        super().__init__(
            name="Ebay",
            base_url="https://www.ebay.com",
            search_path="/sch/i.html?_nkw=",
        )

    def parse_search_results(self, html: str) -> List[BaseModel]:
        """Parse eBay search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("li.s-item"):
            try:
                title_elem = item.select_one("h3.s-item__title")
                title = title_elem.text.strip() if title_elem else "No Title"
                url = item.select_one("a.s-item__link")["href"]
                price_elem = item.select_one("span.s-item__price")
                price = (
                    float(re.sub(r"[^\d.]", "", price_elem.text.strip()))
                    if price_elem
                    else None
                )
                results.append({"title": title, "price": price, "url": url})
            except Exception as e:
                logger.error(f"Error parsing eBay result: {str(e)}")
        return results


class BestBuy(Platform):
    def __init__(self):
        super().__init__(
            name="BestBuy",
            base_url="https://www.bestbuy.com",
            search_path="/site/searchpage.jsp?st=",
        )

    def parse_search_results(self, html: str) -> List[BaseModel]:
        """Parse Best Buy search results"""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("li.sku-item"):
            try:
                title_elem = item.select_one("h4.sku-header")
                title = title_elem.text.strip() if title_elem else "No Title"
                url = urljoin(self.base_url, item.select_one("a.image-link")["href"])
                price_elem = item.select_one("div.priceView-customer-price span")
                price = (
                    float(re.sub(r"[^\d.]", "", price_elem.text.strip()))
                    if price_elem
                    else None
                )
                results.append({"title": title, "price": price, "url": url})
            except Exception as e:
                logger.error(f"Error parsing Best Buy result: {str(e)}")
        return results


class ProductSearchTool:
    """Tool for searching products across platforms"""

    def __init__(self):
        self.platforms = [Amazon(), Ebay(), BestBuy()]

    def _search_platform(self, platform: Platform, query: str) -> List[BaseModel]:
        """Search a single platform"""
        try:
            url = platform.get_search_url(query)
            html = platform.make_request(url)
            return platform.parse_search_results(html)
        except Exception as e:
            logger.error(f"Error searching {platform.name}: {str(e)}")
            return []

    def search_products(self, query: str) -> str:
        """Search products across all platforms"""
        all_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_platform = {
                executor.submit(self._search_platform, p, query): p
                for p in self.platforms
            }
            for future in concurrent.futures.as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(
                        f"Error getting results from {platform.name}: {str(e)}"
                    )
        return json.dumps(all_results)


class ProductComparisonTool:
    """Tool for comparing and analyzing product search results"""

    def __init__(self, model):
        self.model = model

    def compare_products(self, search_results: str) -> str:
        """Use HuggingFace model for analysis"""
        try:
            prompt = f"Analyze the following product listings: {search_results}"
            response = self.model(prompt)
            return response
        except Exception as e:
            logger.error(f"Error comparing products: {str(e)}")
            return f"Error comparing products: {str(e)}"


class ProductSearchAgent:
    def __init__(self):
        """Initialize the product search agent with HuggingFace API model"""
        self.model = pipeline(
            "text-generation", model="gpt2"
        )  # HuggingFace GPT-2 model
        self.search_tool = ProductSearchTool()
        self.comparison_tool = ProductComparisonTool(self.model)

    def search_and_compare(self, query: str) -> str:
        """Search for products and compare them using HuggingFace model"""
        # First, search for products
        search_results = self.search_tool.search_products(query)

        # Now, use the model to compare products
        comparison_result = self.comparison_tool.compare_products(search_results)

        return comparison_result


if __name__ == "__main__":
    # Initialize the Product Search Agent
    agent = ProductSearchAgent()

    # Perform a search query
    query = "gaming laptop under $1500"
    result = agent.search_and_compare(query)

    # Print the result
    print(f"Comparison result for '{query}':\n{result}")
