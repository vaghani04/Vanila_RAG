import streamlit as st
import asyncio
from rag_ingestion_pipeline import rag
import logging

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.web.bootstrap").setLevel(logging.ERROR)

def get_or_create_loop():
    if "loop" not in st.session_state or st.session_state.loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state.loop = loop
        asyncio.set_event_loop(loop)
    return st.session_state.loop

async def run_rag(query):
    return await rag(query)

async def process_query(query):
    response = await run_rag(query)
    st.write("Response:", response["response"])
    image_urls = response["img_urls"]
    st.write(image_urls)
    if image_urls:
        st.subheader("Related Images:")
        for url in image_urls:
            st.image(url, caption=url)

st.title("Simple RAG Chatbot")
query = st.text_input("Enter your query:")

if st.button("Get Response"):
    loop = get_or_create_loop()
    try:
        loop.run_until_complete(process_query(query))
    except Exception as e:
        st.error(f"An error occurred: {e}")