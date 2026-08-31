# src/embeddings/embeddings_pipeline.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ==================================================
# CONFIG
# ==================================================

CHUNKS_FOLDER   = "outputs/chunks"
OUTPUT_FOLDER   = "outputs/embeddings"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 64


# ==================================================
# LOAD ALL CHUNKS
# ==================================================

def load_all_chunks(chunks_folder: str) -> list:

    all_chunks = []

    for filename in sorted(os.listdir(chunks_folder)):

        if not filename.endswith("_chunks.txt"):
            continue

        filepath = os.path.join(chunks_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        blocks = raw.split("=" * 100)

        for block in blocks:

            block = block.strip()

            if not block:
                continue

            lines      = block.split("\n")
            chunk_id   = None
            title      = None
            text_lines = []

            for line in lines:

                if line.startswith("CHUNK ID :"):
                    chunk_id = int(line.replace("CHUNK ID :", "").strip())

                elif line.startswith("TITLE :"):
                    title = line.replace("TITLE :", "").strip()

                else:
                    text_lines.append(line)

            text = "\n".join(text_lines).strip()

            if text:
                all_chunks.append(
                    {
                        "source"   : filename.replace("_chunks.txt", ""),
                        "chunk_id" : chunk_id,
                        "title"    : title,
                        "text"     : text
                    }
                )

    print(f"Total chunks loaded: {len(all_chunks)}")

    return all_chunks


# ==================================================
# GENERATE EMBEDDINGS
# ==================================================

def generate_embeddings(chunks: list, model_name: str) -> np.ndarray:

    print(f"\nLoading model: {model_name}")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


# ==================================================
# CREATE VECTOR DATABASE
# ==================================================

def create_vector_database(chunks: list):

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ── Generate embeddings ──────────────────────
    embeddings = generate_embeddings(chunks, MODEL_NAME)

    # ── Build FAISS index ────────────────────────
    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)   # Inner product (cosine, since normalized)
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal} vectors")

    # ── Save FAISS index ─────────────────────────
    index_path = os.path.join(OUTPUT_FOLDER, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index -> {index_path}")

    # ── Save metadata (chunks without embeddings) ─
    metadata_path = os.path.join(OUTPUT_FOLDER, "chunks_metadata.pkl")

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved metadata    -> {metadata_path}")

    # ── Save embeddings as numpy array ───────────
    embeddings_path = os.path.join(OUTPUT_FOLDER, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings  -> {embeddings_path}")

    return index, chunks, embeddings


# ==================================================
# LOAD VECTOR DATABASE
# ==================================================

def load_vector_database():

    index_path    = os.path.join(OUTPUT_FOLDER, "faiss_index.bin")
    metadata_path = os.path.join(OUTPUT_FOLDER, "chunks_metadata.pkl")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded FAISS index: {index.ntotal} vectors")
    print(f"Loaded metadata   : {len(chunks)} chunks")

    return index, chunks


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    chunks = load_all_chunks(CHUNKS_FOLDER)

    index, chunks, embeddings = create_vector_database(chunks)

    print("\n===================================")
    print("EMBEDDING COMPLETED")
    print("===================================")
    print(f"Total Vectors : {index.ntotal}")
    print(f"Embedding Dim : {EMBEDDING_DIM}")
    print(f"Outputs saved in: {OUTPUT_FOLDER}")