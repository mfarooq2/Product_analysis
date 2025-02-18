import unittest
from unittest.mock import patch
from agent1 import ProductSearchAgent, ProductComparisonTool, ProductSearchTool
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
    @patch('agent1.ProductComparisonTool.__init__')
    def setUp(self):
        self.comparison_tool = ProductComparisonTool()
        self.search_results = ""

    def test_compare_products_empty_string(self):
        with self.assertRaises(ValueError):
            self.comparison_tool.compare_products(self.search_results)


class TestProductSearchTool(unittest.TestCase):
    @patch('agent1.ProductSearchTool.__init__')
    def setUp(self):
        self.search_tool = ProductSearchTool()
        self.query = "test query"

    def test_search_products(self):
        result = self.search_tool.search_products(self.query)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

if __name__ == '__main__':
    unittest.main()
