import streamlit as st
import asyncio
from rag_ingestion_pipeline import rag

def run_rag_sync(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(rag(query))
    loop.close()
    return result

st.title("Simple RAG Chatbot")

query = st.text_input("Enter your query:")

if query:
    response = run_rag_sync(query)
    st.write("Response:", response)