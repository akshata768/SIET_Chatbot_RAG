
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. LOAD PDF
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("No readable text found in the PDF")

    return full_text


# 2. CREATE CHUNKS
def create_chunks(full_text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    while start < len(full_text):

        end = start + chunk_size

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. BUILD VECTORS
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("Cannot create embeddings from empty chunks")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(
        embed_vectors,
        dtype="float32"
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])

    index.add(embed_vectors)

    return embed_model, index


# 4. LOAD BASE
def load_base(pdf_path="01_College_Information.pdf"):

    full_text = load_pdf(pdf_path)

    chunks = create_chunks(full_text)

    embed_model, index = build_vectors(chunks)

    return embed_model, chunks, index


# 5. RETRIEVE CONTENT
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if num_chunks < 1:
        raise ValueError("At least one chunk must be requested")

    query_vector = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    sim_scores, vec_indices = index.search(
        query_vector,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:

        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. MAIN PROGRAM
if __name__ == "__main__":

    print("Loading PDF...")

    embed_model, chunks, index = load_base()

    print("PDF loaded successfully!")

    while True:

        question = input("\nAsk your question (type exit to stop): ")

        if question.lower() == "exit":
            break

        result = retrieve_content(
            question,
            embed_model,
            chunks,
            index
        )

        print("\nRetrieved Content:")
        print(result)
