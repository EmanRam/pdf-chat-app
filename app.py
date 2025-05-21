import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


# 1. Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# 2. Chunk the text
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

# 3. Create vector store
def get_vectorstore(text_chunks):
    model = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=model)
    return vectorstore

# 4. Create conversational chain
def get_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(
        openai_api_key=api_key, 
        openai_api_base="https://openrouter.ai/api/v1",
        model="openai/gpt-3.5-turbo",
        temperature=0
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# 5. Handle user query and update chat history
def handle_userinput(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if i % 2 == 0 else "assistant"):
            st.markdown(message.content)

# MAIN
def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.title("📚 Chat with Multiple PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input box
    user_input = st.chat_input("Ask a question about your documents:")
    if user_input and st.session_state.conversation:
        handle_userinput(user_input)

    with st.sidebar:
        st.markdown("### 📥 Upload PDFs")
        st.markdown("Upload one or more PDFs, then click **Start**.")

        pdf_docs = st.file_uploader(
            label="📄 Select PDFs",
            accept_multiple_files=True,
            type=["pdf"],
            help="Supports multiple files"
        )

        if pdf_docs:
            st.markdown("#### 📝 Uploaded Files")
            for file in pdf_docs:
                st.markdown(f"✅ `{file.name}`")

        st.divider()

        st.markdown("### ⚙️ Process")
        process_button = st.button("🚀 Start", use_container_width=True)

        if process_button:
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
            else:
                with st.spinner("🔍 Reading & indexing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                    st.success("✅ Done! Ask me anything.")



if __name__ == '__main__':
    main()
