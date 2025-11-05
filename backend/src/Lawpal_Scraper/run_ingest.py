import os
import json
from pathlib import Path
from scraper.gov_scraper import crawl_index as crawl_gov
from scraper.court_scraper import crawl_index as crawl_court
from preprocess.text_cleaner import clean_text, content_hash
from embeddings.chunker import chunk_text
from embeddings.embedder import embed_texts
from indexer.faiss_indexer import FaissIndexer

# --- configuration ---
DATA_SOURCES = [
    # # Central Acts and legal updates
    # "https://www.indiacode.nic.in/",
    
    # # Supreme Court judgments
    # "https://main.sci.gov.in/judgments",
    
    # # Bills and amendments
    "https://www.prsindia.org/billtrack"
]

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.pkl"
OUTPUT_JSON = Path("ai_backend/processed_legal_data.json")

def update_processed_json(new_data):
    """Append or update data inside processed_legal_data.json"""
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    else:
        old_data = []

    # Add new records without duplicating same URLs
    existing_urls = {item.get("url") for item in old_data if "url" in item}
    merged_data = old_data + [a for a in new_data if a.get("url") not in existing_urls]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Synced {len(new_data)} new records into processed_legal_data.json")

def main():
    print("Starting LawPal scraper pipeline...")

    # 1. Crawl sources
    all_articles = []
    for src in DATA_SOURCES:
        print(f"\nFetching from {src} ...")
        if "court" in src:
            articles = crawl_court(src)
        else:
            articles = crawl_gov(src)
        all_articles.extend(articles)
        print(f"✓ Got {len(articles)} articles from {src}")

    if not all_articles:
        print("No new articles found.")
        return

    # 2. Process and chunk
    all_chunks = []
    metadatas = []
    for url, article in all_articles:
        text = clean_text(article["text"])
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": url,
                "title": article["title"],
                "chunk_id": i,
                "content_hash": content_hash(chunk),
            })
            all_chunks.append(chunk)

    print(f"\nPrepared {len(all_chunks)} chunks total")

    # 3. Embed
    vectors = embed_texts(all_chunks)
    print("✓ Generated embeddings")

    # 4. Index with FAISS
    dim = len(vectors[0])
    indexer = FaissIndexer(dim)
    indexer.add(vectors, metadatas)
    print("✓ FAISS index updated and saved")

    # 5. Update main dataset
    simplified_articles = [
        {"url": url, "title": article["title"], "text": article["text"]}
        for url, article in all_articles
    ]
    update_processed_json(simplified_articles)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
