import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

from langchain_community.llms import Ollama


def chunk_text(text, chunk_size = 500, overlap = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk  = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


st.set_page_config(page_title="Document question and answer", layout = "wide")
st.title("Document Question and answer 1")
st.write("App is running successfully")

uploaded_file = st.file_uploader("upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    extracted_text = ''
    for doc in documents:
        extracted_text += doc.page_content
    
    chunks = chunk_text(extracted_text)

    st.success(f"PDF is split into {len(chunks)} chunks")

    chunk_documents = []
    
    for chunk in chunks:
        chunk_documents.append(Document(page_content=chunk)) 


    embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vectorstore=Chroma.from_documents(documents = chunk_documents,embedding = embeddings)

    llm = Ollama(model="mistral")

    st.success("embeddings created and stored")

    st.subheader("ask a question about the document")
    user_question = st.text_input("enter your question")

    if user_question:
        retrieved_docs = vectorstore.similarity_search(user_question,k=3)

        st.success("top relevant chunks are retrieved.")

        for i, doc in enumerate(retrieved_docs):
            st.write(f" Result {i+1}")
            st.write(doc.page_content)

        context = ""
        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
        prompt=f"""
        Answer the question based only on the context below.
        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        answer = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(answer)
        
        
    
    st.subheader("sample chunks")
    st.write("chunk 1:")
    st.write(chunks[0])

    if len(chunks) > 1:
        st.write("chunk 2:")
        st.write(chunks[1])

    st.success("pdf loaded successfully")

    st.subheader("extracted text 1")
    st.write(documents[0].page_content)

    os.remove(temp_path)
