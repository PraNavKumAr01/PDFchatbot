import sys
sys.path.append('/Users/pranav/anaconda3/lib/python3.11/site-packages')

# IMPORTING DEPENDENCIES

import random
from apikey import apikey
import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import GooglePalm
from htmlTemplates import css, bot_template, user_template

os.environ['GOOGLE_API_KEY'] = apikey

# LOADING THE FILE INTO CHUNKS

def get_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    pages = text_splitter.split_text(text)
    return pages

# EMBEDDING THE LOADED PDF

def embed_pages(pages):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    store = Chroma.from_texts(pages,embeddings, collection_name='pdfs', persist_directory='db')
    return store

# CREATING RETRIEVAL QA CHAIN

def create_retrieval_qa_chain(store):
    if st.session_state.QAchain is None:
        llm = GooglePalm(temperature = 0.8)
        memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
        QAchain = ConversationalRetrievalChain.from_llm(
            #llm=local_llm,
            llm = llm,
            retriever=store.as_retriever(),
            memory = memory,
        )
        st.session_state.QAchain = QAchain

    return st.session_state.QAchain


# List of possible filler phrases
filler_phrases = [
    "Well,",
    "Actually,",
    "So,",
    "Interestingly,",
]

# List of possible assistance statements
assistance_statements = [
    "How can I help you further?",
    "Is there anything else I can assist you with?",
    "Do you need any further assistance?",
    "Can I help you with anything else?",
    "Feel free to ask more questions!",
]

# Function to generate conversational bot responses
def generate_conversational_response(prev_response, include_assistance=True):
    response = random.choice(filler_phrases) + " " + prev_response.lower()
    
    # Randomly include assistance statement every 2-3 responses
    if include_assistance and random.choice([True, False]):
        assistance = random.choice(assistance_statements)
        response += "<br><br>" + " " + assistance

    return response.capitalize()

def handle_userInput(user_question):
    response = st.session_state.QAchain({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    include_assistance = True  # Include assistance statement initially

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Apply natural language generation to bot responses
            bot_response = generate_conversational_response(message.content, include_assistance)

            # Check if the response is too short and ask clarifying question
            if len(bot_response) < 30:
                bot_response = "Could you provide more details or context about your question? That would help me assist you better."

            st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)
            
            # Toggle assistance statement inclusion every 2-3 responses
            include_assistance = not include_assistance and i % random.randint(2, 3) == 0





# APP FRAMEWORK

def run_app(QAchain):
    prompt = st.text_input("Lets Talk Business")
    if st.button('Enter'):
        if prompt:
            response = QAchain(prompt)
            st.write(response)

def main():

    st.set_page_config(page_title = "Chat with your PDFs", page_icon = ":books:")

    st.write(css, unsafe_allow_html = True)

    if "QAchain" not in st.session_state:
        st.session_state.QAchain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Testing some things")
    user_question = st.text_input("Ask your question")
    if user_question:
        handle_userInput(user_question) 

    with st.sidebar:
        st.subheader("Upload your files")
        pdf_docs = st.file_uploader('Choose your file', type =['pdf'], accept_multiple_files = True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                raw_text = get_text(pdf_docs) 
                text_chunks = get_chunks(raw_text)
                store = embed_pages(text_chunks)
                st.session_state.QAchain = create_retrieval_qa_chain(store)


if __name__ == "__main__":
    main()