
import streamlit as st
import faiss
import numpy as np
import os

from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Load PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))

    mypages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable text found in PDF")

    return full_text


# 2. Create Chunks
def create_chunks(full_text, chunksize=500, overlap=100):
    if chunksize <= 0 or overlap < 0 or overlap >= chunksize:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunksize

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. Build Vectors
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("No chunks available")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(
        embed_vectors, dtype=np.float32
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(
        embed_vectors.shape[1]
    )

    index.add(embed_vectors)

    return index, embed_model


# 4. Load Base
def load_base(pdf_path="01_College_Information.pdf"):

    full_text = load_pdf(pdf_path)

    chunks = create_chunks(full_text)

    index, embed_model = build_vectors(chunks)

    return embed_model, chunks, index


# 5. Retrieve Content
def retrieve_content(
    question, embed_model, chunks, index, num_chunks=3
):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Number of chunks must be at least 1")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed, dtype=np.float32
    )

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. Ask RAG
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("Groq API key is missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    prompt = f"""
You are a professional AI assistant.

Answer using ONLY the document background below.
Do not invent or assume facts.
If information is not available, say Information Missing.

Document Background:
{background}

User Question:
{question}

Answer:
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# STREAMLIT APPLICATION
st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓"
)

st.title("🎓 College RAG Assistant")
st.write("Ask questions based on your college PDF.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your college PDF",
    type=["pdf"]
)

# API key
api_key = st.text_input(
    "Enter Groq API Key",
    type="password"
)

# Question
question = st.text_input(
    "Ask your question"
)

if st.button("Get Answer"):

    if uploaded_file is None:
        st.error("Please upload a PDF file.")

    elif not api_key:
        st.error("Please enter your Groq API key.")

    elif not question.strip():
        st.error("Please enter a question.")

    else:
        try:
            with st.spinner("Reading PDF and creating vectors..."):

                # Save uploaded PDF temporarily
                pdf_path = "uploaded_college.pdf"

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load PDF and build vector database
                embed_model, chunks, index = load_base(pdf_path)

            with st.spinner("Finding the answer..."):

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key
                )

            st.success("Answer generated!")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
