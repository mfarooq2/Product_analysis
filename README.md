# Product Analysis - E-commerce Product Image Processor

## Overview

This project is a web application that leverages the Groq and OpenAI APIs to enhance e-commerce product listings. It automates the process of generating engaging marketing content and image variations directly from product images.

## Features

-   **AI-Powered Content Generation**: Utilizes advanced AI models from Groq to produce creative captions, optimized titles, and compelling product descriptions.
-   **Image Variation**: Employs OpenAI's DALL-E models to generate diverse image variations, offering a range of visual options for the same product.
-   **User-Friendly Interface**: Built with Gradio, it provides an intuitive platform for users to upload images and receive instant AI-generated marketing outputs.

## How It Works

1.  **Image Upload**: Users upload a product image through the Gradio interface.
2.  **AI Processing**: The application processes the image using the following steps:
    -   **Caption Generation**: The `generate_product_details` function uses Groq's models to create a descriptive caption for the product.
    -   **Title Optimization**: The `optimize_title` function uses Groq's models to optimize the generated title for e-commerce use.
    -   **Description Generation**: The `generate_product_description` function utilizes Groq models to generate a compelling product description.
    -   **Image Variation**: The `create_image_variation` function uses DALL-E to generate image variations, providing diverse visual options.
    -   **Image generation from variations**: the function `create_images_from_variations` generates new images based on the product description variations.
3.  **Output**: The application returns a caption, title, product description and all the images to the user
4. **Gallery display** the gallery will display:
    -   First half: Image variations based on your original image
    -   Second half: New images generated based on the product description

## Dependencies

The application relies on the following libraries:

-   `openai`: For generating image variations.
-   `groq`: For content generation and title optimization.
- `matplotlib`: For image manipulation
-   `numpy`: For numerical operations.
- `pandas`: For data manipulation
-   `gradio`: For creating the user interface.
-   `torch`: For image handling.

Ensure these are installed (`pip install -r requirements.txt`).

## API Keys

-   **OPENAI_API_KEY**: Required for image variation.
-   **GROQ_API_KEY**: Required for content generation.

Set these keys as environment variables.

## Usage

1.  Ensure you have the necessary API keys set up.
2.  Upload a product image via the Gradio interface.
3.  Click "Process Image" to generate the AI-driven marketing content and image variations.
4.  Utilize the generated output for your e-commerce platform.

## Possible improvements

- Adding a progress bar to show the users the time each process is taking to complete
- Adding a history section for the user to check their previous inputs and the outputs.
- Add more text and images capabilities to the app.
- Add the option to download the generated images.
- Improve the UX and the UI of the app.

## License

This project is open-source and available for modification under the MIT License.

## Contact

For more information, questions, or feature requests, please [open an issue](https://github.com/mfarooq2/Product_analysis).
