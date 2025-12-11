import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

class BiomedicalDocumentRetriever:

    def __init__(self, temp_dir="data/pdfs"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.europepmc_search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search_europepmc(self, query: str, start_year=None, end_year=None, max_results=10):
        q = query
        if start_year and end_year:
            q += f" AND PUB_YEAR:[{start_year} TO {end_year}] AND OPEN_ACCESS:Y"
        params = {
            "query": q,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
            "synonym": "true"
        }
        try:
            r = requests.get(self.europepmc_search_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data.get("resultList", {}).get("result", [])
        except Exception as e:
            print(f"✗ Europe PMC search error: {e}")
            return []

    def extract_pdf_link(self, entry):
        urls = entry.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for u in urls:
            if u.get("documentStyle") == "pdf":
                return u.get("url")
        # fallback using PMCID
        pmcid = entry.get("pmcid")
        if pmcid:
            return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        return None

    def download_pdf(self, title: str, url: str):
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        out_path = self.temp_dir / f"{safe_title}.pdf"
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                return str(out_path)
        except Exception as e:
            print(f"✗ Error downloading PDF {title}: {e}")
        return None

    def save_abstract(self, entry, filename: str):
        abstract = entry.get("abstractText")
        if not abstract:
            return None
        safe_title = "".join(c if c.isalnum() else "_" for c in filename)[:50]
        out_path = self.temp_dir / f"{safe_title}.txt"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {entry.get('title')}\n")
                f.write(f"Authors: {entry.get('authorString')}\n")
                f.write(f"Year: {entry.get('pubYear')}\n")
                f.write(f"Journal: {entry.get('journalTitle')}\n\n")
                f.write(abstract)
            return str(out_path)
        except Exception as e:
            print(f"✗ Error saving abstract {filename}: {e}")
            return None

    def retrieve_documents(self, query, start_year=None, end_year=None,
                       max_papers=5, download_pdfs=True, save_abstracts=True):

        entries = self.search_europepmc(query, start_year, end_year, max_results=max_papers*3)
        results = []
        downloaded_count = 0

        # Use tqdm with total=max_papers so it stops at max_papers
        with tqdm(total=max_papers, desc="Downloading papers") as pbar:
            for entry in entries:
                if downloaded_count >= max_papers:
                    break

                title = entry.get("title", "paper")
                pdf_path = None
                abstract_path = None

                pdf_url = self.extract_pdf_link(entry)
                if download_pdfs and pdf_url:
                    pdf_path = self.download_pdf(title, pdf_url)
                else:
                    continue  # skip entries without PDFs

                if save_abstracts:
                    abstract_path = self.save_abstract(entry, title)

                results.append({
                    "title": title,
                    "year": entry.get("pubYear"),
                    "authors": entry.get("authorString"),
                    "journal": entry.get("journalTitle"),
                    "pmcid": entry.get("pmcid"),
                    "doi": entry.get("doi"),
                    "pdf_path": pdf_path,
                    "abstract_path": abstract_path
                })

                if pdf_path:
                    downloaded_count += 1
                    pbar.update(1)  # update progress bar only for successful downloads

        return {
            "documents": results,
            "stats": {
                "retrieved": len(results),
                "pdf_downloaded": sum(1 for r in results if r["pdf_path"]),
                "abstracts_saved": sum(1 for r in results if r["abstract_path"])
            }
        }


