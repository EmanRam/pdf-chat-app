import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    model = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=model)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="openai/gpt-3.5-turbo",
        temperature=1
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_input):
    with st.chat_message("user"):
        st.markdown(user_input)
    response = st.session_state.conversation({'question': user_input})
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    st.session_state.chat_history = response['chat_history']

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")

    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide"
    )

    # Header centered at top of main page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📚 PDF Chat Assistant")
        st.markdown("##### Upload your documents and start chatting!")
    st.divider()

    # Initialize session state variables if not exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Document Management moved to sidebar exactly as original
    with st.sidebar:
        st.header("📁 Document Management")
        with st.container():
            st.subheader("Upload PDFs")
            pdf_docs = st.file_uploader(
                "Choose PDF files",
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload one or more PDF documents"
            )

            if pdf_docs:
                st.success(f"✅ {len(pdf_docs)} file(s) uploaded")

                with st.expander("📋 File Details"):
                    for i, pdf in enumerate(pdf_docs, 1):
                        st.write(f"{i}. {pdf.name}")
                        st.write(f"   Size: {pdf.size / 1024:.1f} KB")

            st.divider()

            if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        progress_text.text("📖 Extracting text...")
                        progress_bar.progress(25)
                        raw_text = get_pdf_text(pdf_docs)

                        progress_text.text("✂️ Creating chunks...")
                        progress_bar.progress(50)
                        text_chunks = get_text_chunks(raw_text)

                        progress_text.text("🧠 Building vector store...")
                        progress_bar.progress(75)
                        vectorstore = get_vectorstore(text_chunks)

                        progress_text.text("🔗 Setting up conversation...")
                        progress_bar.progress(90)
                        conversation = get_conversation_chain(vectorstore, api_key)

                        st.session_state.conversation = conversation
                        st.session_state.chat_history = []

                        progress_bar.progress(100)
                        progress_text.text("✅ Complete!")

                    st.balloons()
                    st.success("🎉 PDFs processed successfully!")
                else:
                    st.warning("⚠️ Please upload PDF files first")

        # Session info and clear chat in sidebar as original
        if st.session_state.conversation:
            st.divider()
            with st.container():
                st.subheader("📊 Session Info")

                message_count = len(st.session_state.chat_history) // 2
                st.metric("Messages Exchanged", message_count)

                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    if hasattr(st.session_state.conversation, 'memory'):
                        st.session_state.conversation.memory.clear()

    # Main page chat interface (same as original)
    st.header("💬 Chat Interface")

    if st.session_state.conversation:
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.chat_history:
                role = "user" if message.type == "human" else "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)

        user_input = st.chat_input("Ask a question about your documents...")
        if user_input and st.session_state.conversation:
            handle_userinput(user_input)

    else:
        st.info("👆 Upload and process your PDF documents to start chatting!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Key Features:**
            - Multi-document support
            - Intelligent Q&A
            - Context-aware responses
            - Conversation memory
            """)
        with col2:
            st.markdown("""
            **🚀 How to get started:**
            1. Upload your PDF files
            2. Click 'Process PDFs'
            3. Start asking questions!
            4. Enjoy smart conversations
            """)

if __name__ == "__main__":
    main()
