import json
from pathlib import Path
from datetime import datetime

METADATA_FILE = Path(__file__).parent / "scraper_metadata.json"

def load_metadata():
    """Load stored metadata (URLs, timestamps, hashes)"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """Write metadata back to file"""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def update_metadata(source_url, article_url, content_hash):
    """Add or update metadata for a scraped article"""
    metadata = load_metadata()
    now = datetime.utcnow().isoformat()

    metadata[article_url] = {
        "source": source_url,
        "content_hash": content_hash,
        "last_updated": now,
    }

    save_metadata(metadata)
    print(f"🗂️ Updated metadata for {article_url}")

def has_changed(article_url, new_hash):
    """Return True if content hash differs from stored one"""
    metadata = load_metadata()
    entry = metadata.get(article_url)
    return not entry or entry.get("content_hash") != new_hash
