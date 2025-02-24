import uuid
from typing import List, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import motor.motor_asyncio
import asyncio
from text_summaries import text_summaries
from image_summaries import image_summaries
from chunking import chunks as texts
from get_images import images

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "summaries"

existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["vanila_rag"]
text_collection = db["text_chunks"]
image_collection = db["image_chunks"]

async def add_text_documents(texts: List[Any], text_summaries: List[str]) -> List[str]:
    doc_ids = [f"text_{str(uuid.uuid4())}" for _ in texts]
    
    for i, text in enumerate(texts):
        if hasattr(text, 'metadata') and hasattr(text.metadata, 'orig_elements'):
            chunk_text = " ".join(
                str(elem.text) for elem in text.metadata.orig_elements 
                if hasattr(elem, 'text') and elem.text
            )
            text_content = chunk_text if chunk_text else str(text)
        else:
            text_content = str(text)
        await text_collection.insert_one({"doc_id": doc_ids[i], "content": text_content})
    
    summary_embeddings = model.encode(text_summaries)
    vectors = []
    for i, embedding in enumerate(summary_embeddings):
        vectors.append({
            "id": doc_ids[i],
            "values": embedding.tolist(),
            "metadata": {"type": "text", "summary": text_summaries[i], "doc_id": doc_ids[i]}
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    
    return doc_ids

async def add_image_documents(images: List[Any], image_summaries: List[str]) -> List[str]:
    img_ids = [f"image_{str(uuid.uuid4())}" for _ in images]
    
    for i, img in enumerate(images):
        image_content = str(img)
        await image_collection.insert_one({"doc_id": img_ids[i], "content": image_content})
    
    summary_embeddings = model.encode(image_summaries)
    vectors = []
    for i, embedding in enumerate(summary_embeddings):
        vectors.append({
            "id": img_ids[i],
            "values": embedding.tolist(),
            "metadata": {"type": "image", "summary": image_summaries[i], "doc_id": img_ids[i]}
        })
    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    
    return img_ids

async def main():
    text_ids = await add_text_documents(texts, text_summaries)
    image_ids = await add_image_documents(images, image_summaries)
    print(f"Text doc_ids: {text_ids}")
    print(f"Image doc_ids: {image_ids}")

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
