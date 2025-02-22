# Product Analysis Tool

This project implements a product search and comparison tool. It leverages web scraping to gather product information from various e-commerce platforms, including Amazon, eBay, and Best Buy. This data is then analyzed and compared using a HuggingFace language model.

## Features

-   **Web Scraping:** Extracts product details from e-commerce websites.
-   **Product Search:** Allows users to search for products across multiple platforms.
-   **Product Comparison:** Employs a HuggingFace model to analyze and compare product listings.
    The model compares various product attributes including descriptions, prices, and reviews.
    It leverages natural language processing techniques to assess the quality and similarity of product features.

## Usage

1.  **Search:** Provide a search query (e.g., "gaming laptop under $1500").
2.  **Comparison:** The tool will return a comparison of the products found.

## Components

-   `agent1.py`: Contains the core logic for web scraping, product searching, and comparison.
-   `Platform` class: Base class for e-commerce platforms.
-   `Amazon`, `Ebay`, `BestBuy` classes: Specific implementations for each platform.
-   `ProductSearchTool` class: Manages searching across platforms.
-   `ProductComparisonTool` class: Handles product comparison using the HuggingFace model.
-   `ProductSearchAgent` class: Orchestrates the entire process.

## Dependencies

-   `requests`: For making HTTP requests.
-   `BeautifulSoup`: For parsing HTML content.
-   `pydantic`: For data validation.
-   `transformers`: For language model integration.
-   `logging`: For logging errors and events.

## Getting Started

To get started, follow these steps:

1. Install the required dependencies:

       pip install -r requirements.txt

2. Run the script using:

       python agent1.py

The script's main function, `search_and_compare`, performs a product search and comparison.
**API Model Configuration**
- To change the default model, you can modify the `model_name` parameter in the main block `if __name__ == "__main__":`
## Contributing

Contributions are welcome! If you find any bugs or want to add new features, feel free to submit a pull request.

## License

This project is licensed under the MIT License.