# RAG-PDF-Question-Answering--local-LLM

A local Retrieval-Augmented Generation (RAG) based Document Q&amp;A application built using Python, Streamlit, LangChain, ChromaDB, and Ollama. The system allows users to upload a PDF and ask questions strictly based on the document content.

- This project is a local Retrieval-Augmented Generation (RAG) based Document Question & Answering system.  
It allows users to upload a PDF document and ask questions, ensuring that answers are generated **only from the provided document**.

The application runs completely locally using **Ollama**, ensuring data privacy and avoiding the use of paid APIs.

# INSTALLATION : 

# Python : 

- Python 3.10 or above  

Check using:

```bash

python --version
```

# Install Ollama from:

https://ollama.com/

Pull the required model:

```
ollama pull mistral
```

Check if it is installed properly using :

```
run ollama mitral
```

# Create Virtual Environment

```
python -m venv venv
```

# Activate Virtual Environment

For Windows

```
.\venv\Scripts\activate
```

# Install Dependencies

```
pip install -r requirements.txt
```

# Running the Application

Run the Streamlit app using:

```
streamlit run app.py
```

The browser opens.

# How It Works (Architecture)

1.User uploads a PDF document.

2.Text is extracted using PyPDFLoader.

3.Text is split into overlapping chunks (500 characters with 50 overlap).

4.Chunks are converted into embeddings using sentence-transformers.

5.Embeddings are stored in a Chroma vector database.

6.User asks a question.

7.Relevant chunks are retrieved and passed to the LLM (Mistral via Ollama).

8. The model generates a context-aware answer.

# Progress :

Streamlit-based web interface:

<img width="1918" height="926" alt="image" src="https://github.com/user-attachments/assets/73ebbf36-0a58-4aa9-86a6-ee61e1509a93" />

# Final Results :

<img width="1918" height="926" alt="image" src="https://github.com/user-attachments/assets/4da4f483-6f1c-4f44-bd9b-c9f16a5f590a" />

