import fitz  # PyMuPDF
from nomic import embed
from pathlib import Path
import os
import spacy
nlp = spacy.load("en_core_web_sm")

CHUNK_SIZE = 500  # tokens/words approximation

def get_pdf_files(directory="data/pdfs"):
    directory = Path(directory)
    files = list(directory.glob("*.pdf"))
    print(f"Found {len(files)} PDF files to embed.")
    return files

def embed_query_text(query: str):
    """Embed natural-language query string using Nomic."""
    from nomic import embed
    result = embed.text([query])
    return result["embeddings"][0]  # return embedding vector

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, max_words=200):
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        sentence_words = sentence.split()

        # If adding this sentence exceeds the chunk size → start new chunk
        if len(current_chunk_words) + len(sentence_words) > max_words:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = sentence_words
        else:
            current_chunk_words.extend(sentence_words)

    # Add leftover words as a chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def embed_pdf_texts(pdf_files):
    all_chunks = []
    chunk_index_tracker = []
    chunk_text_tracker = []  

    print(f"Preparing chunks for {len(pdf_files)} PDFs ...")

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        all_chunks.extend(chunks)
        chunk_index_tracker.append(len(chunks))
        chunk_text_tracker.append(chunks)  # store text chunks per document

    print(f"Total chunks prepared: {len(all_chunks)}")

    # Embed all chunks at once
    print("Generating embeddings...")
    response = embed.text(
        texts=all_chunks,
        model="nomic-embed-text-v1"
    )

    embeddings = response["embeddings"]

    # Re-group per PDF (text + embedding)
    grouped_results = []
    index = 0

    for pdf_path, chunk_count, chunk_texts in zip(pdf_files, chunk_index_tracker, chunk_text_tracker):
        pdf_chunks = []

        for i in range(chunk_count):
            pdf_chunks.append({
                "chunk_id": i,
                "text": chunk_texts[i],              # store chunk text
                "embedding": embeddings[index + i]   # store embedding
            })

        grouped_results.append({
            "file": str(pdf_path),
            "num_chunks": chunk_count,
            "chunks": pdf_chunks                    # each is {chunk_id, text, embedding}
        })

        index += chunk_count

    return grouped_results

