import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
import speech_recognition as sr
from gtts import gTTS
import os

groq_api_key = "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"

# Define a prompt template to improve interaction
def format_prompt(user_query):
    return f"""
    You are a friendly and helpful healthcare assistant. Your role is to provide guidance and general information based on the uploaded documents and the user's query. Please be aware of the following:

    1. **Caution:** The information provided is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns or questions.
    2. **Context:** Use the information from the uploaded documents to inform your responses. If the documents do not cover the specific query, provide general information based on commonly accepted knowledge.
    3. **Tone:** Be empathetic, professional, and clear. Avoid medical jargon and provide information in an easy-to-understand manner.

    User's Query: {user_query}

    Assistant:
    """

def main():
    st.set_page_config(page_title="🏥 Healthcare Chatbot", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        .main { background-color: #f5f5f5; }
        .sidebar { background-color: #f0f0f0; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Healthcare Chatbot")
    
    st.image('img1.jpeg')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processcomplete" not in st.session_state:
        st.session_state.processcomplete = None
    
    # Define the path to the fixed location document
    document_path = 'healthcare_document.pdf'
    
    process = st.button("Process")
    
    if process:
        if not groq_api_key:
            st.error("API key is missing. Please check your configuration.")
            st.stop()
        if not os.path.exists(document_path):
            st.warning("Document not found. Please check the path.")
            st.stop()
        
        with st.spinner("Processing Document..."):
            doc = get_file_text(document_path)
            st.success("Document Loaded.")
            text_chunks = get_text_chunks(doc)
            st.success("Text Chunks Created.")
            vector_store = get_vectorstore(text_chunks)
            st.success("Vector Store Created.")
        
        st.session_state.conversation = get_conversation_chain(vector_store, groq_api_key)
        st.session_state.processcomplete = True
    
    input_type = st.radio("Choose input type:", ('Text', 'Voice'))
    
    if st.session_state.processcomplete:
        if input_type == 'Text':
            user_question = st.text_input("Your Query", placeholder="Type your question here...")
        else:
            user_question = get_voice_input()
        
        if user_question:
            with st.spinner("Working on your request..."):
                handle_user_input(user_question)

def get_file_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        text += get_pdf_text(file_path)
    elif file_path.endswith(".docx"):
        text += get_docx_text(file_path)
    return text

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(docx_file):
    all_text = []
    doc = docx.Document(docx_file)
    for para in doc.paragraphs:
        all_text.append(para.text)
    text = ''.join(all_text)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=4096, chunk_overlap=300, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vectorstore, api_keys):
    llm = ChatGroq(groq_api_key=api_keys, model_name='Llama3-70b-8192', temperature=0.5)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(user_question):
    formatted_prompt = format_prompt(user_question)
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': formatted_prompt})
    st.session_state.chat_history = response['chat_history']

    response_text = response['chat_history'][-1].content

    # Display text response
    st.markdown(f"**Assistant:** {response_text}")

    # Convert text to speech and play it
    play_response_voice(response_text)

def play_response_voice(response_text):
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")
    st.audio("response.mp3", format='audio/mp3')

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        user_question = recognizer.recognize_google(audio)
        st.write(f"You said: {user_question}")
        return user_question
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

if __name__ == '__main__':
    main()
