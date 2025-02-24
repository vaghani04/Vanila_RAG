from chunking import chunks
import os
from dotenv import load_dotenv
from google import generativeai
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in the .env file.")

generativeai.configure(api_key=GOOGLE_API_KEY)

model = generativeai.GenerativeModel("gemini-1.5-flash")

prompt_template = """
You are an AI assistant analyzing research paper images.
Describe the image in detail, focusing on graphs, bar plots, and structural elements.

- Identify the type of graph (bar, line, pie, etc.).
- Mention key labels, trends, and data points.
- If text is present, summarize it.
- Provide a structured explanation.

Avoid unnecessary commentary and provide a concise yet informative summary.
"""

def summarize_base64_image(image_b64):
    try:
        image_data = {
            "mime_type": "image/jpeg",
            "data": image_b64
        }

        response = model.generate_content([prompt_template, image_data])

        return response.text.strip() if response and response.text else "No description available."
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)

image_summary_file = "image_summaries.json"
if os.path.exists(image_summary_file):
    with open(image_summary_file, "r", encoding="utf-8") as f:
        image_summaries = json.load(f)
else:
    image_summaries = [summarize_base64_image(image) for image in images]
    with open(image_summary_file, "w", encoding="utf-8") as f:
        json.dump(image_summaries, f, ensure_ascii=False, indent=4)
