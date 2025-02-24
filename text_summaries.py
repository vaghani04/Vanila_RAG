import os
from dotenv import load_dotenv
from google import generativeai
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in the .env file.")

generativeai.configure(api_key=GOOGLE_API_KEY)

model = generativeai.GenerativeModel("gemini-pro")

prompt_template = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

def summarize_text(text_chunk):
    prompt = prompt_template.format(element=text_chunk)
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No summary available."
    except Exception as e:
        return f"Error during generation: {str(e)}"

from chunking import chunks as texts

# text_summaries = [summarize_text(text) for text in texts]

# for i, summary in enumerate(text_summaries, 1):
#     print(f"Summary {i}: {summary}")

from chunking import chunks as texts

summary_file = "text_summaries.json"
if os.path.exists(summary_file):
    with open(summary_file, "r", encoding="utf-8") as f:
        text_summaries = json.load(f)
else:
    text_summaries = [summarize_text(text) for text in texts]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(text_summaries, f, ensure_ascii=False, indent=4)
