import uuid
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import motor.motor_asyncio
import asyncio

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "summaries"
index = pc.Index(index_name)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["vanila_rag"]
text_collection = db["text_chunks"]
image_collection = db["image_chunks"]

import base64
import requests

def fetch_image_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a valid response
    return base64.b64encode(response.content).decode('utf-8')


async def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query])[0].tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    matches = results["matches"]
    retrieved_data = []
    img_urls = []
    for match in matches:
        doc_id = match["metadata"]["doc_id"]
        doc_type = match["metadata"]["type"]
        summary = match["metadata"]["summary"]
        score = match["score"]
        if doc_type == "text":
            chunk_doc = await text_collection.find_one({"doc_id": doc_id})
            content = chunk_doc["content"] if chunk_doc else None
        elif doc_type == "image":
            chunk_doc = await image_collection.find_one({"doc_id": doc_id})
            image_url = chunk_doc["img_url"] if chunk_doc else None
            img_urls.append(image_url)
            if image_url:
                content = fetch_image_base64(image_url)
        else:
            chunk_doc = None
        # content = chunk_doc["content"] if chunk_doc else None
        retrieved_data.append({
            "type": doc_type,
            "summary": summary,
            "doc_id": doc_id,
            "content": content,
            "score": score
        })
    return {"retrieved_data": retrieved_data, "img_urls": img_urls}

# async def main():
#     query = "What is the attention mechanism & how it is applied in the transformer architecture?"
#     retrieved_chunks = await retrieve_chunks(query, top_k=5)
#     for item in retrieved_chunks["retrieved_data"]:
#         print(f"Type: {item['type']}")
#         print(f"Doc ID: {item['doc_id']}")
#         print(f"Summary: {item['summary']}")
#         print(f"Content: {item['content']}")
#         print(f"Score: {item['score']}")
#         print("---")

# if __name__ == "__main__":
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         loop.run_until_complete(main())
#     finally:
#         pending = asyncio.all_tasks(loop)
#         if pending:
#             loop.run_until_complete(asyncio.gather(*pending))
#         client.close()
#         loop.close()