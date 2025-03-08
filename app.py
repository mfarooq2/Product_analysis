import os
import base64
import requests
import io
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image
from openai import OpenAI
from groq import Groq
import time

# Initialize API Keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Validate API Keys
if not OPENAI_API_KEY or not GROQ_API_KEY:
    raise ValueError(
        "Missing API Keys. Please set OPENAI_API_KEY and GROQ_API_KEY in environment variables."
    )


# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to encode PIL Image to base64
def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Function to generate product details from image
def generate_product_details(image, groq_api_key=None):
    try:
        # Initialize the client with the API key from environment if not provided
        groq_client = Groq(api_key=groq_api_key or GROQ_API_KEY)

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert to base64
        base64_image = pil_to_base64(image)

        chat_completion = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Create a creative and engaging caption for this product image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"


# Function to optimize title
def optimize_title(title, groq_api_key=None):
    try:
        groq_client = Groq(api_key=groq_api_key or GROQ_API_KEY)
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate an optimized version of this e-commerce title: {title} in under 250 characters",
                }
            ],
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# Function to generate product description
def generate_product_description(title, groq_api_key=None):
    try:
        groq_client = Groq(api_key=groq_api_key or GROQ_API_KEY)
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a catchy, marketable, and compelling description for this title: {title} in under 500 characters",
                }
            ],
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# Function to generate product description variations


def generate_description_variations(description, groq_api_key=None, n=3):
    try:
        groq_client = Groq(api_key=groq_api_key or GROQ_API_KEY)
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a variation of this product description with minor changes like color, size, or attributes while keeping the meaning intact in less than 900 characters: {description}",
                }
            ],
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# Function to create image variations
def create_image_variation(image, openai_api_key=None, n=2):
    try:
        openai_client = OpenAI(api_key=openai_api_key or OPENAI_API_KEY)

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Save temporarily
        temp_path = "temp_image.png"
        image.save(temp_path)

        with open(temp_path, "rb") as image_file:
            response = openai_client.images.create_variation(
                image=image_file, model="dall-e-2", n=n, size="256x256"
            )

        variation_urls = [data.url for data in response.data]

        # Download and convert URLs to images
        images = []
        for url in variation_urls:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            images.append(np.array(img))

        return images
    except Exception as e:
        return [f"Error: {str(e)}"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Function to create images from captions
def create_images_from_variations(
    product_description_variations, openai_api_key=None, n=2
):
    openai_client = OpenAI(api_key=openai_api_key or OPENAI_API_KEY)
    response = openai_client.images.generate(
        prompt=product_description_variations, model="dall-e-2", n=n, size="256x256"
    )
    variation_urls = [data.url for data in response.data]

    # Download and convert URLs to images
    images = []
    for url in variation_urls:
        try:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            images.append(np.array(img))
        except:
            images.append(np.zeros((256, 256, 3), dtype=np.uint8))
    return images


# Function to process everything
def process_product_image(image, num_variations=2):
    # Use environment variables for API keys
    openai_api_key = OPENAI_API_KEY
    groq_api_key = GROQ_API_KEY
    
    with gr.Blocks() as progress_block:
        gr.Markdown("## Progress")
        progress_bar = gr.Textbox(label="Progress", interactive=False)
        
    # Create the history to store data
    history_data = {
        "caption": "",
        "title": "",
        "description": "",
        "image": None,
        "variations": [],
        "description_images": [],
    }
    
    # Generate product details
    caption = generate_product_details(image, groq_api_key)
    history_data["caption"] = caption
    progress_bar.value = "Processing"
    
    # Optimize title
    optimized_title = optimize_title(caption, groq_api_key)
    history_data["title"] = optimized_title
    progress_bar.value = "Processing"

    # Generate product description
    product_description = generate_product_description(optimized_title, groq_api_key)
    history_data["description"] = product_description
    progress_bar.value = "Processing"

    # Create image variations - use num_variations
    image_variations = create_image_variation(image, openai_api_key, n=num_variations)
    history_data["variations"] = image_variations
    progress_bar.value = "Processing"
    
    product_description_variations = generate_description_variations(
        product_description, groq_api_key
    )
    progress_bar.value = "Processing"

    # Create images from description - use num_variations
    description_images = create_images_from_variations(
        product_description_variations, openai_api_key, n=num_variations
    )
    history_data["description_images"] = description_images
    progress_bar.value = "Processing"

    # Combine all images for gallery - total will be 2*num_variations
    all_images = image_variations + description_images
    history_data["all_images"] = all_images
    progress_bar.value = "Done"
    
    add_to_history(history_data)
    
    #add the image to the history
    history_data["image"]=image

    return caption, optimized_title, product_description, image, all_images


# Create Gradio Interface
with gr.Blocks(title="E-commerce Product Image Processor") as demo:
    gr.Markdown("# E-commerce Product Image Processor")
    gr.Markdown(
        "Upload a product image and get AI-generated marketing content and image variations"
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Input")
            image_input = gr.Image(label="Upload Product Image", type="pil")

            # Updated slider with minimum 2, step size 2, default 2
            num_variations = gr.Slider(
                minimum=2,
                maximum=10,
                step=2,
                value=2,
                label="Number of image variations (per type)",
            )

            process_button = gr.Button("Process Image")
        with gr.Column():
            gr.Markdown("## Output")
            original_image = gr.Image(label="Original Image")
            caption_output = gr.Textbox(label="Generated Caption")
            title_output = gr.Textbox(label="Optimized Title")
            description_output = gr.Textbox(label="Product Description")
            text_input = gr.Textbox(label="Text Input")

        # Gallery will display 2*num_variations images with one image per column
        image_gallery = gr.Gallery(
            label="Image Variations & Generated Images",
            show_label=True,
            columns=2,
            rows=1,
            height="600",
        )

    gr.Markdown("## How to use")
    gr.Markdown(
        """
    ### Instructions
    1. Upload a product image
    2. Select number of image variations (2-10, in steps of 2)
    3. Click 'Process Image' to generate marketing content and image variations
    4. Use the generated content for your e-commerce listings
    
    **Note**: This application requires OPENAI_API_KEY and GROQ_API_KEY environment variables to be set.
    
    **The gallery will display**:
    - First half: Image variations based on your original image
    - Second half: New images generated based on the product description
    """
    )
    gr.Markdown("## Gallery")
    gr.Markdown("### Instructions")
    gr.Markdown("""
    #### Note
    This application requires OPENAI_API_KEY and GROQ_API_KEY environment variables to be set.
    """)
    gr.Markdown("""
    **The gallery will display**:
    - First half: Image variations based on your original image
    - Second half: New images generated based on the product description
    """)
    history_section = gr.Dataframe(label="History", interactive=False)

    def update_history():
        return get_history()
    
    process_button.click(
        fn=process_product_image,
        inputs=[image_input, num_variations],
        outputs=[
            caption_output,
            title_output,
            description_output,
            original_image,
            image_gallery,
        ],
    )
    
    process_button.click(update_history, outputs=[history_section])
    
    download_button = gr.Button("Download Images")
    
    def download_images(images):
        for image in images:
            img = Image.fromarray(image)
            img.save(f"image_{images.index(image)}.png")
        return "Images Downloaded"
    download_button.click(download_images, inputs=[image_gallery], outputs=[])

    gr.Markdown("## How to use")
    gr.Markdown(
        """
    1. Upload a product image
    2. Select number of image variations (2-10, in steps of 2)
    3. Click 'Process Image' to generate marketing content and image variations
    4. Use the generated content for your e-commerce listings
    
    Note: This application requires OPENAI_API_KEY and GROQ_API_KEY environment variables to be set.
    
    The gallery will display:
    - First half: Image variations based on your original image
    - Second half: New images generated based on the product description
    """
    )

# Function to store data in the history
history = []

def add_to_history(data):
    history.append(data)
    
def get_history():
    return history
    
if __name__ == "__main__":
    demo.launch()