
import faiss
import numpy as np

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------
# 1. FIND PDF FILES
# --------------------------------

def find_pdf_files(data_folder):

    folder = Path(data_folder)

    pdf_files = list(folder.rglob("*.pdf"))

    return pdf_files


# --------------------------------
# 2. LOAD PDF FILES
# --------------------------------

def load_pdf(data_folder):

    all_pages = []

    pdf_files = find_pdf_files(data_folder)

    if not pdf_files:

        raise ValueError(
            "No PDF files found in the project folder."
        )

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
# 3. CREATE TEXT CHUNKS
# --------------------------------

def create_chunks(
    pages,
    chunk_size=800,
    overlap=100
):

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
# 4. CREATE EMBEDDINGS
# --------------------------------

def create_embeddings(chunks, model):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    return embeddings


# --------------------------------
# 5. CREATE FAISS DATABASE
# --------------------------------

def create_vector_database(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------
# 6. LOAD COMPLETE RAG SYSTEM
# --------------------------------

def load_vector_store(data_folder):

    print("Loading PDF files...")

    pages = load_pdf(data_folder)

    if not pages:

        raise ValueError(
            "PDF files contain no readable text."
        )

    print("Creating text chunks...")

    chunks = create_chunks(pages)

    if not chunks:

        raise ValueError(
            "No text chunks created."
        )

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
# 7. RETRIEVE INFORMATION
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
# 8. GENERATE ANSWER USING GROQ
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

    Answer the user's question using the information
    available in the supplied context.

    Explain in simple and clear English.

    If the answer is not available in the context,
    say that the uploaded documents do not contain
    enough information to answer the question.

    CONTEXT:

    {context}

    USER QUESTION:

    {question}

    ANSWER:
    """

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

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
