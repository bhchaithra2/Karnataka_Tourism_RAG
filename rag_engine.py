import os
import faiss
import numpy as np

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------
# 1. LOAD PDF FILES
# --------------------------------

def load_pdf(data_folder):

    all_pages = []

    pdf_files = Path(data_folder).glob("*.pdf")

    for pdf_path in pdf_files:

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            if text.strip():

                all_pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number + 1
                })

    return all_pages


# --------------------------------
# 2. TEXT CHUNKING
# --------------------------------

def create_chunks(pages, chunk_size=500, overlap=50):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# --------------------------------
# 3. GENERATE EMBEDDINGS
# --------------------------------

def create_embeddings(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    return embeddings


# --------------------------------
# 4. CREATE FAISS DATABASE
# --------------------------------

def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------
# 5. LOAD COMPLETE RAG SYSTEM
# --------------------------------

def load_vector_store(data_folder):

    print("Loading PDF files...")

    pages = load_pdf(data_folder)

    if not pages:

        raise ValueError(
            "No readable PDF text found in Data_files."
        )

    print("Creating text chunks...")

    chunks = create_chunks(pages)

    if not chunks:

        raise ValueError("No text chunks created.")

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = create_embeddings(
        chunks,
        model
    )

    print("Creating FAISS database...")

    index = create_vector_database(
        embeddings
    )

    print("RAG system is ready!")

    return index, chunks, model


# --------------------------------
# 6. RETRIEVE RELEVANT INFORMATION
# --------------------------------

def retrieve_context(
    question,
    index,
    chunks,
    model,
    top_k=4
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for i in indices[0]:

        if i != -1:

            results.append(chunks[i])

    return results


# --------------------------------
# 7. GENERATE ANSWER USING GROQ LLM
# --------------------------------

def generate_answer(
    question,
    retrieved_data,
    api_key
):

    client = Groq(
        api_key=api_key
    )

    context = ""

    for item in retrieved_data:

        context += (
            "\nSource: " + item["source"]
            + "\nPage: " + str(item["page"])
            + "\nInformation: " + item["text"]
            + "\n"
        )

    prompt = f"""
    You are a helpful Karnataka Tourism Assistant.

    Answer the user's question using only the
    information provided in the context.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the uploaded documents.

    Give simple, clear and useful answers.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3,

        max_tokens=512
    )

    answer = response.choices[0].message.content

    return answer
