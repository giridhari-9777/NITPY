# src/vectorstore/chroma_store.py

import os
import sys
import pickle
import numpy as np
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIG
# ==================================================

CHROMA_DB_PATH        = "./chroma_db"
MEDICAL_COLLECTION    = "medical_rag"        # Cancer knowledge base
HISTORY_COLLECTION    = "conversation_history" # Doctor-Patient chat history
EMBEDDINGS_FOLDER     = "outputs/embeddings"
MODEL_NAME            = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE            = 500


# ==================================================
# CHROMA CLIENT
# ==================================================

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


# ==================================================
# COLLECTION 1 — MEDICAL KNOWLEDGE BASE
# ==================================================

def get_medical_collection(client):
    return client.get_or_create_collection(
        name     = MEDICAL_COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


# ==================================================
# COLLECTION 2 — CONVERSATION HISTORY
# ==================================================

def get_history_collection(client):
    return client.get_or_create_collection(
        name     = HISTORY_COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


# ==================================================
# INSERT MEDICAL CHUNKS
# ==================================================

def insert_chunks(chunks: list, embeddings: np.ndarray):

    client     = get_chroma_client()
    collection = get_medical_collection(client)

    existing = collection.count()

    if existing > 0:
        print(f"Medical KB already has {existing} records. Skipping.")
        return collection

    print(f"\nInserting {len(chunks)} chunks into Medical KB...")

    for start in range(0, len(chunks), BATCH_SIZE):

        end        = min(start + BATCH_SIZE, len(chunks))
        batch      = chunks[start:end]
        batch_embs = embeddings[start:end]

        ids       = [f"id_{i}" for i in range(start, end)]
        documents = [chunk["text"] for chunk in batch]
        metadatas = [
            {
                "source"   : str(chunk.get("source",   "unknown")),
                "title"    : str(chunk.get("title",    "no_title")),
                "chunk_id" : str(chunk.get("chunk_id", i))
            }
            for i, chunk in enumerate(batch)
        ]

        collection.add(
            ids        = ids,
            documents  = documents,
            embeddings = batch_embs.tolist(),
            metadatas  = metadatas
        )

        print(f"  Inserted {start} → {end}")

    print(f"Medical KB ready! Total: {collection.count()} records")
    return collection


# ==================================================
# MEDICAL KNOWLEDGE SEARCH
# ==================================================

def medical_search(
    query_embedding : np.ndarray,
    k               : int = 10
) -> list:

    client     = get_chroma_client()
    collection = get_medical_collection(client)

    result = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results        = k,
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []

    for i in range(len(result["ids"][0])):
        chunks.append({
            "idx"    : result["ids"][0][i],
            "text"   : result["documents"][0][i],
            "source" : result["metadatas"][0][i].get("source", "unknown"),
            "title"  : result["metadatas"][0][i].get("title",  "no_title"),
            "score"  : 1 - result["distances"][0][i]
        })

    return chunks


# ==================================================
# SAVE CONVERSATION TURN
# (Doctor-Patient Q&A history)
# ==================================================

def save_conversation_turn(
    session_id  : str,
    role        : str,    # "patient" or "doctor"
    message     : str,
    model       : SentenceTransformer,
    metadata    : dict = {}
):

    client     = get_chroma_client()
    collection = get_history_collection(client)

    # Embed the message
    embedding = model.encode(
        message,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    turn_id   = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    collection.add(
        ids        = [turn_id],
        documents  = [message],
        embeddings = [embedding.tolist()],
        metadatas  = [{
            "session_id" : session_id,
            "role"       : role,
            "timestamp"  : timestamp,
            **metadata
        }]
    )

    return turn_id


# ==================================================
# GET FULL SESSION HISTORY
# ==================================================

def get_session_history(session_id: str) -> list:

    client     = get_chroma_client()
    collection = get_history_collection(client)

    result = collection.get(
        where   = {"session_id": session_id},
        include = ["documents", "metadatas"]
    )

    if not result["ids"]:
        return []

    # Sort by timestamp
    turns = []

    for i in range(len(result["ids"])):
        turns.append({
            "id"        : result["ids"][i],
            "message"   : result["documents"][i],
            "role"      : result["metadatas"][i].get("role", "unknown"),
            "timestamp" : result["metadatas"][i].get("timestamp", ""),
            "session_id": result["metadatas"][i].get("session_id", "")
        })

    turns.sort(key=lambda x: x["timestamp"])

    return turns


# ==================================================
# SEARCH SIMILAR PAST QUESTIONS
# (Find if patient asked similar before)
# ==================================================

def search_similar_history(
    query       : str,
    session_id  : str,
    model       : SentenceTransformer,
    k           : int = 3
) -> list:

    client     = get_chroma_client()
    collection = get_history_collection(client)

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        query,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    result = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results        = min(k, collection.count()),
        where            = {"session_id": session_id},
        include          = ["documents", "metadatas", "distances"]
    )

    similar = []

    for i in range(len(result["ids"][0])):
        similar.append({
            "message"   : result["documents"][0][i],
            "role"      : result["metadatas"][0][i].get("role", "unknown"),
            "timestamp" : result["metadatas"][0][i].get("timestamp", ""),
            "score"     : 1 - result["distances"][0][i]
        })

    return similar


# ==================================================
# FORMAT HISTORY FOR PROMPT
# ==================================================

def format_history_for_prompt(history: list, last_n: int = 6) -> str:

    if not history:
        return "No previous conversation."

    # Take last N turns
    recent = history[-last_n:]

    lines = []

    for turn in recent:

        role    = "Patient" if turn["role"] == "patient" else "Doctor"
        message = turn["message"]
        lines.append(f"{role}: {message}")

    return "\n".join(lines)


# ==================================================
# CHROMA STORE CLASS
# ==================================================

class ChromaStore:

    def __init__(self):

        print("\nInitializing ChromaStore...")

        self.client      = get_chroma_client()
        self.medical_col = get_medical_collection(self.client)
        self.history_col = get_history_collection(self.client)
        self.model       = SentenceTransformer(MODEL_NAME)

        print(f"Medical KB     : {self.medical_col.count()} records")
        print(f"History DB     : {self.history_col.count()} records")
        print("ChromaStore ready!\n")


    # ── Load medical knowledge ──────────────────

    def load_and_insert(self):

        metadata_path   = os.path.join(EMBEDDINGS_FOLDER, "chunks_metadata.pkl")
        embeddings_path = os.path.join(EMBEDDINGS_FOLDER, "embeddings.npy")

        with open(metadata_path, "rb") as f:
            chunks = pickle.load(f)

        embeddings = np.load(embeddings_path)

        insert_chunks(chunks, embeddings)


    # ── Search medical knowledge ────────────────

    def search(self, query: str, k: int = 10) -> list:

        query_embedding = self.model.encode(
            query,
            normalize_embeddings = True,
            convert_to_numpy     = True
        )

        return medical_search(query_embedding, k=k)


    # ── Save a conversation turn ────────────────

    def save_turn(
        self,
        session_id : str,
        role       : str,
        message    : str,
        metadata   : dict = {}
    ):
        return save_conversation_turn(
            session_id = session_id,
            role       = role,
            message    = message,
            model      = self.model,
            metadata   = metadata
        )


    # ── Get full session history ────────────────

    def get_history(self, session_id: str) -> list:
        return get_session_history(session_id)


    # ── Format history for LLM prompt ──────────

    def format_history(
        self,
        session_id : str,
        last_n     : int = 6
    ) -> str:
        history = get_session_history(session_id)
        return format_history_for_prompt(history, last_n)


    # ── Find similar past questions ─────────────

    def find_similar_questions(
        self,
        query      : str,
        session_id : str,
        k          : int = 3
    ) -> list:
        return search_similar_history(
            query      = query,
            session_id = session_id,
            model      = self.model,
            k          = k
        )


    # ── Stats ───────────────────────────────────

    def stats(self):

        print(f"\nChromaDB Stats")
        print(f"==============")
        print(f"Medical KB : {self.medical_col.count()} records")
        print(f"History DB : {self.history_col.count()} sessions")
        print(f"Path       : {CHROMA_DB_PATH}")


# ==================================================
# MAIN — Test Doctor-Patient Flow
# ==================================================

if __name__ == "__main__":

    store      = ChromaStore()
    session_id = "patient_001"

    # Load medical KB if empty
    if store.medical_col.count() == 0:
        store.load_and_insert()

    store.stats()

    print("\n" + "="*60)
    print("SIMULATING DOCTOR-PATIENT CONVERSATION")
    print("="*60)

    # Simulate conversation turns
    conversation = [
        ("patient", "I have been coughing for 3 months and losing weight."),
        ("doctor",  "These symptoms can sometimes be associated with lung conditions including cancer. I recommend immediate imaging tests."),
        ("patient", "What are the symptoms of lung cancer?"),
        ("doctor",  "Common symptoms include persistent cough, chest pain, shortness of breath, and unexplained weight loss."),
        ("patient", "What is the survival rate for stage 2 lung cancer?"),
    ]

    for role, message in conversation:
        store.save_turn(session_id, role, message)
        print(f"\n{'Patient' if role == 'patient' else 'Doctor'}: {message}")

    # Get full history
    print("\n" + "="*60)
    print("FULL SESSION HISTORY")
    print("="*60)
    history = store.get_history(session_id)
    for turn in history:
        role = "Patient" if turn["role"] == "patient" else "Doctor"
        print(f"{role} [{turn['timestamp'][:19]}]: {turn['message']}")

    # Format for LLM
    print("\n" + "="*60)
    print("FORMATTED FOR LLM PROMPT (last 6 turns)")
    print("="*60)
    print(store.format_history(session_id))

    # Search medical KB
    print("\n" + "="*60)
    print("MEDICAL KB SEARCH")
    print("="*60)
    results = store.search("survival rate lung cancer stage 2", k=3)
    for i, chunk in enumerate(results):
        print(f"\n[{i+1}] Source : {chunk['source']}")
        print(f"     Score  : {chunk['score']:.4f}")
        print(f"     Preview: {chunk['text'][:200]}...")