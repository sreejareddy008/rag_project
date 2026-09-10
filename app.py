from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
reader = PdfReader("documents/ai_notes.pdf")

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"\nTotal Chunks: {len(chunks)}")

print("\nLoading Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded!")


client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="ai_notes"
)

print("\nCreating Embeddings...")

embeddings = model.encode(chunks)

print("Embeddings Shape:", embeddings.shape)



print("\nAdding documents to ChromaDB...")

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("ChromaDB Index Created")
print("Vectors Stored:", collection.count())

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

print("\nTop Results\n")

documents = results["documents"][0]
distances = results["distances"][0]

for rank, (document, distance) in enumerate(
    zip(documents, distances),
    start=1
):

    print("=" * 60)
    print(f"RESULT {rank}")
    print("=" * 60)

    print(document)

    print("\nDistance:", distance)
    print()