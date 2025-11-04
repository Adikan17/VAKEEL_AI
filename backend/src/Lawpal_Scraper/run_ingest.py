import os
from scraper.gov_scraper import crawl_index as crawl_gov
from scraper.court_scraper import crawl_index as crawl_court
from preprocess.text_cleaner import clean_text, content_hash
from embeddings.chunker import chunk_text
from embeddings.embedder import embed_texts
from indexer.faiss_indexer import FaissIndexer

# --- configuration ---
DATA_SOURCES = [
    # Central Acts and legal updates
    "https://www.indiacode.nic.in/",
    
    # Supreme Court judgments
    "https://main.sci.gov.in/judgments",
    
    # Bills and amendments
    "https://www.prsindia.org/billtrack"
]


INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.pkl"

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

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
