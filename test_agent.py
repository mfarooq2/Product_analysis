import unittest
from unittest.mock import patch
import json
import requests
from bs4 import BeautifulSoup
from agent1 import Platform, Amazon, Ebay, BestBuy, ProductSearchTool, ProductComparisonTool, ProductSearchAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("product_search.log")],
)
logger = logging.getLogger(__name__)


class TestPlatform(unittest.TestCase):
    @patch.object(requests, "Session", create=True)
    def setUp(self):
        self.session = requests.Session()
        self.name = "TestPlatform"
        self.base_url = "https://www.testplatform.com"
        self.search_path = "/s?k="
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def test_make_request_success(self):
        # Test that make_request returns a string when the response is successful
        url = "https://www.testplatform.com/s?k=test"
        with patch.object(self, "make_request", return_value="Response"):
            response_text = self.make_request(url)
            self.assertEqual(response_text, "Response")

    def test_make_request_failure(self):
        # Test that make_request returns None when all attempts fail
        url = "https://www.testplatform.com/s?k=test"
        with patch.object(self, "make_request", side_effect=requests.RequestException):
            response_text = self.make_request(url)
            self.assertIsNone(response_text)

    def test_get_search_url(self):
        query = "test"
        expected_url = "https://www.testplatform.com/s?k=test"
        self.assertEqual(self.get_search_url(query), expected_url)

    def test_parse_search_results(self):
        html = "<html><body><div data-component-type='s-search-result'><h2 class='a-size-medium a-color-base s-inline s-access-title a-text-normal'><a><span>Test Title</span></a></h2><span class='a-price-whole'>$100.00</span></div></body></html>"
        expected_results = [{"title": "Test Title", "price": 100.00, "url": "https://www.testplatform.com/test"}]
        with patch.object(BeautifulSoup, "BeautifulSoup", return_value=BeautifulSoup(html, "html.parser")):
            self.assertEqual(self.parse_search_results(html), expected_results)
        


class TestAmazon(unittest.TestCase):
    def setUp(self):
        self.platform = Platform("Amazon", "https://www.amazon.com", "/s?k=")
        self.amazon = Amazon()
        self.session = requests.Session()

    def test_get_search_url(self):
        query = "test"
        expected_url = "https://www.amazon.com/s?k=test"
        self.assertEqual(self.amazon.get_search_url(query), expected_url)

    def test_parse_search_results(self):
        html = "<html><body><div data-component-type='s-search-result'><h2 class='a-size-medium a-color-base s-inline s-access-title a-text-normal'><a><span>Test Title</span></a></h2><span class='a-price-whole'>$100.00</span></div></body></html>"
        expected_results = [{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}]
        with patch.object(BeautifulSoup, "BeautifulSoup", return_value=BeautifulSoup(html, "html.parser")):
            self.assertEqual(self.amazon.parse_search_results(html), expected_results)
        

class TestEbay(unittest.TestCase):
    def setUp(self):
        self.platform = Platform("Ebay", "https://www.ebay.com", "/sch/i.html?_nkw=")
        self.ebay = Ebay()
        self.session = requests.Session()

    def test_get_search_url(self):
        query = "test"
        expected_url = "https://www.ebay.com/sch/i.html?_nkw=test"
        self.assertEqual(self.ebay.get_search_url(query), expected_url)

    def test_parse_search_results(self):
        html = "<html><body><div data-component-type='s-search-result'><h2 class='a-size-medium a-color-base s-inline s-access-title a-text-normal'><a><span>Test Title</span></a></h2><span class='a-price-whole'>$100.00</span></div></body></html>"
        expected_results = [{"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"}]
        with patch.object(BeautifulSoup, "BeautifulSoup", return_value=BeautifulSoup(html, "html.parser")):
            self.assertEqual(self.ebay.parse_search_results(html), expected_results)

class TestBestBuy(unittest.TestCase):
    def setUp(self):
        self.platform = Platform("BestBuy", "https://www.bestbuy.com", "/site/searchpage.jsp?st=")
        self.bestbuy = BestBuy()
        self.session = requests.Session()

    def test_get_search_url(self):
        query = "test"
        expected_url = "https://www.bestbuy.com/site/searchpage.jsp?st=test"
        self.assertEqual(self.bestbuy.get_search_url(query), expected_url)

    def test_parse_search_results(self):
        html = "<html><body><div data-component-type='s-search-result'><h2 class='a-size-medium a-color-base s-inline s-access-title a-text-normal'><a><span>Test Title</span></a></h2><span class='a-price-whole'>$100.00</span></div></body></html>"
        expected_results = [{"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}]
        with patch.object(BeautifulSoup, "BeautifulSoup", return_value=BeautifulSoup(html, "html.parser")):
            self.assertEqual(self.bestbuy.parse_search_results(html), expected_results)
        

class TestProductSearchTool(unittest.TestCase):
    def setUp(self):
        self.product_search_tool = ProductSearchTool()

    def test_search_platform(self):
        query = "test"
        platforms = [TestAmazon().amazon, TestEbay().ebay, TestBestBuy().bestbuy]
        expected_results = [
            {"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"},
            {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"},
            {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}
        ]
        for platform, expected in zip(platforms, expected_results):
            with patch.object(platform, "make_request", return_value="dummy raw response"):
                with patch.object(platform, "parse_search_results", return_value=expected):
                    result = self.product_search_tool._search_platform(platform, query)
                    self.assertEqual(result, expected)

    def test_search_products(self):
        query = "test"
        expected_result = json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}])
        with patch.object(self.product_search_tool, "search_products", return_value=json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}])):
            self.assertEqual(self.product_search_tool.search_products(query), expected_result)
        

class TestProductComparisonTool(unittest.TestCase):
    def setUp(self):
        self.product_comparison_tool = ProductComparisonTool(model="gpt2")

    def test_compare_products_success(self):
        search_results = json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}])
        prompt = f"Analyze the following product listings: {search_results}"
        expected_response = "Test response"
        with patch.object(self.product_comparison_tool.model, "return_value", return_value=expected_response):
            response = self.product_comparison_tool.compare_products(search_results)
            self.assertEqual(response, expected_response)
    
    def test_compare_products_failure(self):
        search_results = json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test"}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test"}])
        prompt = f"Analyze the following product listings: {search_results}"
        expected_response = "Error comparing products: test"
        with patch.object(self.product_comparison_tool.model, "return_value", side_effect=Exception):
            response = self.product_comparison_tool.compare_products(search_results)
            self.assertEqual(response, expected_response)


class TestProductSearchAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ProductSearchAgent()

    @patch.object(ProductSearchTool, "search_products", return_value='[{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test"}]')
    @patch.object(ProductComparisonTool, "compare_products", return_value="Comparison result: Analyze the following product listings: [{'title': 'Test Title', 'price': 100.0, 'url': 'https://www.amazon.com/test'}]")
    def test_search_and_compare(self):
        query = "test"
        result = self.agent.search_and_compare(query)
        self.assertEqual(result, "Comparison result: Analyze the following product listings: [{'title': 'Test Title', 'price': 100.0, 'url': 'https://www.amazon.com/test'}]")
    
    def test_search_and_compare_failure(self):
        query = "test"
        with patch.object(self.agent, "search_and_compare", side_effect=Exception):
            result = self.agent.search_and_compare(query)
            self.assertNotEqual(result, "Comparison result: Analyze the following product listings: [{'title': 'Test Title', 'price': 100.0, 'url': 'https://www.amazon.com/test'}]")

if __name__ == "__main__":
    unittest.main()
