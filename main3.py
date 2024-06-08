import gradio as gr
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import ollama
import os

# Load the environment variables from the .env file
load_dotenv()

# Install the Google Generative AI SDK
# $ pip install google-generativeai

import google.generativeai as genai

# Configure the Google Generative AI client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the Google Generative AI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

import requests
from requests.exceptions import ReadTimeout

# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict() 
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            return vectorstore.as_retriever()
        except ReadTimeout:
            retry_count += 1
            continue

    # If maximum retries reached
    print("Maximum retries reached. Unable to fetch the URL.")
    return None

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(url, question):
    retriever = load_and_retrieve_docs(url)
    
    if retriever is None:
        return "Error: Maximum retries reached. Unable to fetch the URL."
    
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(formatted_prompt)
    
    return response.text.strip()

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text", "text"],
    outputs="text",
    title="CHAT WITH URL",
    description="Enter a URL and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()
