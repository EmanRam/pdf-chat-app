# 📚 Chat with Multiple PDFs

An AI-powered web app that lets you upload multiple PDF documents and interact with their content using natural language questions. Built with **Streamlit**, **LangChain**, **FAISS**, and **OpenAI** via **OpenRouter**.

---

## ✨ Features

- 📄 Upload and process **multiple PDF files**
- 🔍 Automatically **extracts, chunks, and embeds** PDF content
- 🧠 Uses **FAISS** for efficient semantic search
- 💬 Chat with your documents using **GPT-3.5 Turbo**
- 🧵 Maintains conversation history using **LangChain memory**
- 🖥️ Clean and intuitive **Streamlit UI**

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/EmanRam/pdf-chat-app.git
cd pdf-chat-app
```

## 2. Install Dependencies

Install required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 3. Set Up Environment Variables

Create a .env file in the root directory and add your OpenRouter API key:

```bash
API_KEY=your_openrouter_api_key_here
```

## 4. Run the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

## 🖼️ UI Overview

### Sidebar
- 📥 Upload PDFs  
- ⚙️ Start document processing

### Main Area
- 📚 Title  
- 💬 Chat interface to ask questions

---

### 📬 Contact

If you have suggestions or questions, feel free to connect via [LinkedIn](https://www.linkedin.com/in/eman-ramzy-6976091b4/).
