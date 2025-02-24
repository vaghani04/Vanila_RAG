import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import motor.motor_asyncio
import asyncio
from rag_retrieval import retrieve_chunks
import google.generativeai as genai
from dataclasses import dataclass
from base64 import b64decode

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("summaries")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["my_database"]
text_collection = db["text_chunks"]
image_collection = db["image_chunks"]

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

@dataclass
class Document:
    page_content: str
    metadata: dict

async def retrieve_documents_as_docs(query, top_k=5):
    retrieved = await retrieve_chunks(query, top_k)
    docs = []
    for doc in retrieved:
        metadata = {
            "id": doc["doc_id"],
            "score": doc["score"],
            "type": doc["type"],
            "summary": doc["summary"]
        }
        docs.append(Document(page_content=doc["content"], metadata=metadata))
    return docs

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc.page_content)
            b64.append(doc.page_content)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    for text_doc in docs_by_type["texts"]:
        context_text += text_doc.page_content + "\n"

    prompt_parts = [
        f"Answer the question based only on the following context, which can include text and images.\nContext:\n{context_text}\nQuestion: {user_question}"
    ]

    for image_b64 in docs_by_type["images"]:
        prompt_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_b64
            }
        })

    return prompt_parts

async def rag(query):
    docs = await retrieve_documents_as_docs(query)
    parsed = parse_docs(docs)
    prompt_parts = build_prompt({"context": parsed, "question": query})
    response = gemini_model.generate_content(prompt_parts)
    return response.text

async def rag_with_sources(query):
    docs = await retrieve_documents_as_docs(query)
    parsed = parse_docs(docs)
    prompt_parts = build_prompt({"context": parsed, "question": query})
    response = gemini_model.generate_content(prompt_parts)
    return {
        "response": response.text,
        "context": parsed
    }

async def main():
    query = "What is the scaled dot product and multi head attention?"
    response = await rag(query)
    print(type(response))

    result = await rag_with_sources(query)
    print("Response:", result["response"])
    for image in result["context"]["images"]:
        print("Image (base64):", image)
        print(f'\n\n HERE IS THE IMAGE: {type(image)}')

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
        client.close()
        loop.close()