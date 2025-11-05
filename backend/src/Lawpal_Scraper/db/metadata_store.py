import json
from pathlib import Path
from datetime import datetime

METADATA_FILE = Path(__file__).resolve().parent / "scraper_metadata.json"

def load_metadata():
    """Load stored metadata (document title, timestamp, hash, etc.)."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """Save metadata dictionary to JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def update_metadata(doc_title, content_hash, document_type, source_url):
    """Save/update metadata for a single document."""
    metadata = load_metadata()
    now = datetime.utcnow().isoformat()

    metadata[doc_title] = {
        "content_hash": content_hash,
        "document_type": document_type,
        "source_url": source_url,
        "last_scraped": now
    }

    save_metadata(metadata)
    print(f"🗂️ Metadata updated for '{doc_title}'")

def has_changed(doc_title, new_hash):
    """Check if document has new content (compares hash)."""
    metadata = load_metadata()
    entry = metadata.get(doc_title)
    return not entry or entry.get("content_hash") != new_hash
