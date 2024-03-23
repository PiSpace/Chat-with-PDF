import streamlit as st
import os
import tempfile
import shelve
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set your Google API Key here. It's recommended to use environment variables for security.
os.environ["GOOGLE_API_KEY"] = "AIzaSyDhaendGvFHUO_YxFPBx4tGu2MbS3k0Aws"
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents and concatenate text from all pages."""
    text = ""
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(pdf.getvalue())
            tmpfile.seek(0)
            pdf_reader = PdfReader(tmpfile.name)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split the extracted text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Embed text chunks using a specified model and save the results in a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversation_chain():
    """Set up and return a conversation chain for answering questions based on the context provided."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, 'Answer is not available in the context'; don't provide the wrong answer.\n\n
    Context: {context}?\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, prompt=prompt)

def load_chat_history():
    """Load previously saved chat history from a shelve database."""
    with shelve.open("chat_history.db") as db:
        return db.get("current_session", [])

def save_chat_history(messages):
    """Save the current chat history to a shelve database for persistence."""
    with shelve.open("chat_history.db") as db:
        db["current_session"] = messages

def main():
    """Main function to run the Streamlit app with UI and performance improvements."""
    st.title("🧠 PDF Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    if "input_key" not in st.session_state:
        st.session_state.input_key = 0  # Initialize input key for clearing input field

    with st.sidebar:
        st.title("Your Chats")
        pdf_docs = st.file_uploader("Upload PDF Files:", accept_multiple_files=True, type=["pdf"])
        if pdf_docs:
            with st.spinner('Processing PDFs...'):  # Provides user feedback during processing
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("PDFs processed. You can now ask questions.")

        if st.button("Reset Chat"):
            st.session_state.messages = []
            save_chat_history(st.session_state.messages)
            st.experimental_rerun()

    display_chat_history()

    user_query = st.text_input("How can I assist?", key=f"user_query_{st.session_state.input_key}")
    if st.button("Send") and user_query:
        with st.spinner("Thinking..."):  # Feedback during bot response generation
            process_user_query(user_query)
        st.session_state.input_key += 1  # Clears the input field after submission

def display_chat_history():
    """Enhanced chat history display with improved UI."""
    st.markdown("### Chat History")
    for msg in st.session_state.messages:
        # Use columns for layout improvements
        col1, col2 = st.columns([1, 15])
        with col1:
            st.markdown(f"{msg['avatar']}")
        with col2:
            st.markdown(f"**{msg['role']}**: {msg['content']}")
def process_user_query(user_query):
    """Process the user's query and generate a response from the chatbot."""
    st.session_state.messages.append({"role": "You", "content": user_query, "avatar": USER_AVATAR})
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_query, top_k=3)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    st.session_state.messages.append({"role": "Bot", "content": response["output_text"], "avatar": BOT_AVATAR})
    save_chat_history(st.session_state.messages)

if __name__ == "__main__":
    main()
