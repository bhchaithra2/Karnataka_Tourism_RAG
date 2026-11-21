
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


#load pdfs
def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    mypages = [i.extract_text() or "" for i in reader.pages]

    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable Text found")

    return full_text


#find pdfs
def find_pdf_files(data_folder):

    folder = Path(data_folder)

    pdf_files = list(folder.rglob("*.pdf"))

    return pdf_files


#chunking
def create_chunks(full_text, chunksize=500, overlap=100):

    chunks = []

    start = 0

    while start < len(full_text):

        end = start + chunksize

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


#embedding
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")

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

    return index, embed_model


#call loadpdf, chunk, embedding
def load_base(data_folder):

    pdf_files = find_pdf_files(data_folder)

    if not pdf_files:
        raise ValueError("No PDF files found")

    full_text = ""

    for pdf_path in pdf_files:

        full_text += load_pdf(pdf_path) + "\n"

    chunks = create_chunks(full_text)

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


#retrieve
def retrieve_content(question, embed_model, chunks, index, num_chunks=4):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one or more chunks")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed,
        dtype="float32"
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


#ask
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=4,
    model_name="openai/gpt-oss-120b"
):

    key = api_key

    if not key:
        raise ValueError("API key missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks=num_chunks
    )

    prompt = """You are a professional Karnataka Tourism AI Assistant.

    Answer the user's question using only the document
    background provided below.

    Explain in simple and clear English.

    Do not invent or assume facts.

    If information is not available, simply say
    information is missing.

    Document Background:
    """ + background + """

    User Question:
    """ + question + """

    Answer:
    """

    client = Groq(api_key=key)

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content
