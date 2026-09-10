from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

client = chromadb.PersistentClient(
    path="data/chroma_db"
)
collection = client.get_collection(
    name="ai_notes"
)

print(
    f"\nVectors available: {collection.count()}"
)

query = input("\nAsk Question: ")

query_embedding = model.encode(
    [query]
)[0]

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

documents = results["documents"][0]
distances = results["distances"][0]

print("\nTop Results\n")

for rank, (document, distance) in enumerate(
    zip(documents, distances),
    start=1
):

    print("=" * 60)
    print(f"RESULT {rank}")
    print("=" * 60)

    print(document)

    print("\nDistance:", distance)