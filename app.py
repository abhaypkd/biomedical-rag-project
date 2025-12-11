from rag.query_parser import QueryParser
from rag.keyword_extractor import KeywordExtractor
from rag.document_retriever import BiomedicalDocumentRetriever
from rag.pdf_embedder import embed_pdf_texts, get_pdf_files, embed_query_text
from rag.faiss_retriever import FaissRetriever
import numpy as np
import json

print("\n--- STEP-1 Query Parsing ---")
query = "What is the effect of aspirin in cardiovascular disease from papers published in 2020-2024?"

parser = QueryParser()
parsed = parser.parse_structured_query(query)
start_year, end_year = parser.parse_timeframe(parsed['timeframe'])
print(parsed, start_year, end_year)

print("\n--- STEP-2 Keyword Extraction ---")
extractor = KeywordExtractor()
keywords = extractor.extract_keywords(
    parsed['research_keyword'] + " " + parsed['intervention'] + " " + parsed['condition']
)
print(keywords)

search_terms = " ".join([
    parsed['research_keyword'],
    parsed['intervention'],
    parsed['condition']
] + keywords[:5]).lower()

print("\nSearch Terms:", search_terms)


print("\n--- STEP-3 Document Retrieval ---")
retriever = BiomedicalDocumentRetriever(temp_dir="data/pdfs")
results = retriever.retrieve_documents(
    query=search_terms,
    start_year=start_year,
    end_year=end_year,
    max_papers=5,
    download_pdfs=True,
    save_abstracts=True
)

docs = results["documents"]
print(f"\nRetrieved {len(docs)} documents\n")

print("\nDownloaded PDFs:")
for i, d in enumerate(docs, 1):
    if d["pdf_path"]:
        print(f"{i}. {d['pdf_path'].split('/')[-1]}")


print("\n--- STEP-4 Embedding PDFs ---")

# Get list of PDF files
pdf_files = get_pdf_files("data/pdfs")

# embed_pdf_texts() now returns full chunk+embedding structure
embeddings_output = embed_pdf_texts(pdf_files)

print(f"Generated embeddings for {len(embeddings_output)} PDFs")

# Save JSON
output_path = "data/pdf_embeddings.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(embeddings_output, f)

print(f"Embeddings saved to {output_path}")


print("\n--- STEP-5 Semantic Search with FAISS ---")
faiss_retriever = FaissRetriever("data/pdf_embeddings.json")
query_embedding = embed_query_text(search_terms)
results = faiss_retriever.search(query_embedding, top_k=5)

print(f"\nTop 5 relevant chunks:")
for i, res in enumerate(results, 1):
    print(f"{i}. PDF: {res['pdf']}")
    print(f"   Chunk ID: {res['chunk_id']}")
    print(f"   Distance: {res['distance']:.4f}")
    print(f"   Chunk Text:\n{res['text'][:350]}...\n")