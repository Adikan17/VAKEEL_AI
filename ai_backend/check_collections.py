from chromadb import PersistentClient

client = PersistentClient(path="chroma_db")
print("✅ ChromaDB connected.")

collections = client.list_collections()
print("\nAvailable collections:")
for c in collections:
    print(" -", c.name)
