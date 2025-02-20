import unittest
from unittest.mock import patch
from app import ProductSearchAgent, ProductComparisonTool, ProductSearchTool, Product
from langchain.agents import initialize_agent, AgentType
from langchain.llms import initialize_llm
from langchain.tools import StructuredTool, ToolType


class TestProductSearchAgent(unittest.TestCase):
    @patch('agent1.ProductSearchAgent.__init__')
    def setUp(self):
        self.agent = ProductSearchAgent()
        self.query = "test query"

    def test_search_and_compare(self):
        result = self.agent.search_and_compare(self.query)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

class TestProductComparisonTool(unittest.TestCase):
    def setUp(self):
        self.product_comparison_tool = ProductComparisonTool(model_identifier="gpt2")

    def test_compare_products_empty_string(self):
        with self.assertRaises(ValueError):
            self.product_comparison_tool.compare_products("")

    def test_compare_products_detailed_empty_list(self):
        products = []
        expected_response = "No products to compare."
        response = compare_products_detailed(products)
        self.assertEqual(response, expected_response)
    
    def test_compare_products_detailed_success(self):
        product1 = Product(title="Test Title", price=100.00, url="https://www.amazon.com/test", specs="Specs", pros=["Pro1", "Pro2"], cons=["Cons1", "Cons2"])
        product2 = Product(title="Test Title", price=100.00, url="https://www.ebay.com/test", specs="Specs", pros=["Pro1", "Pro2"], cons=["Cons1", "Cons2"])
        products = [product1, product2]
        expected_response = "Best product: Product(title='Test Title', price=100.0, url='https://www.amazon.com/test', specs='Specs', pros=['Pro1', 'Pro2'], cons=['Cons1', 'Cons2'])\nComparison: Product: Test Title\nSpecs: Specs\nPros: Pro1, Pro2\nCons: Cons1, Cons2\nProduct: Test Title\nSpecs: Specs\nPros: Pro1, Pro2\nCons: Cons1, Cons2"
        response = compare_products_detailed(products)
        self.assertEqual(response, expected_response)

class TestProductSearchTool(unittest.TestCase):
    def setUp(self):
        self.product_search_tool = ProductSearchTool()

    def test_search_products_empty(self):
        
        self.assertEqual(self.product_search_tool.search_products(""), "[]")
        
        
    def test_search_products(self):
        query = "test"
        expected_result = json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}])
        with patch.object(self.product_search_tool, "search_products", return_value=json.dumps([{"title": "Test Title", "price": 100.00, "url": "https://www.amazon.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}, {"title": "Test Title", "price": 100.00, "url": "https://www.ebay.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}, {"title": "Test Title", "price": 100.00, "url": "https://www.bestbuy.com/test", "specs": "Specs", "pros": ["Pro1", "Pro2"], "cons": ["Cons1", "Cons2"]}])):
            self.assertEqual(self.product_search_tool.search_products(query), expected_result)
        
    

if __name__ == '__main__':
    unittest.main()
