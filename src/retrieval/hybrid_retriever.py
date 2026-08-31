# src/retrieval/hybrid_retriever.py

import os
import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==================================================
# CONFIG
# ==================================================

EMBEDDINGS_FOLDER = "outputs/embeddings"
MODEL_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DENSE       = 20
TOP_K_BM25        = 20
TOP_K_FINAL       = 10


# ==================================================
# LOAD VECTOR DATABASE
# ==================================================

def load_vector_database():

    index_path    = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.bin")
    metadata_path = os.path.join(EMBEDDINGS_FOLDER, "chunks_metadata.pkl")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"FAISS index    : {index.ntotal} vectors")
    print(f"Chunks loaded  : {len(chunks)}")

    return index, chunks


# ==================================================
# DENSE RETRIEVER (FAISS)
# ==================================================

def dense_retrieve(
    query  : str,
    index,
    chunks : list,
    model  : SentenceTransformer,
    top_k  : int = TOP_K_DENSE
) -> list:

    query_embedding = model.encode(
        [query],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk               = chunks[idx].copy()
        chunk["dense_score"] = float(score)
        chunk["idx"]         = int(idx)

        results.append(chunk)

    return results


# ==================================================
# SPARSE RETRIEVER (BM25)
# ==================================================

def build_bm25_index(chunks: list):

    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_corpus)


def sparse_retrieve(
    query  : str,
    bm25,
    chunks : list,
    top_k  : int = TOP_K_BM25
) -> list:

    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        chunk              = chunks[idx].copy()
        chunk["bm25_score"] = float(scores[idx])
        chunk["idx"]        = int(idx)

        results.append(chunk)

    return results


# ==================================================
# RECIPROCAL RANK FUSION
# ==================================================

def reciprocal_rank_fusion(
    dense_results  : list,
    sparse_results : list,
    k              : int = 60
) -> list:

    scores = {}

    for rank, chunk in enumerate(dense_results):
        idx = chunk["idx"]
        if idx not in scores:
            scores[idx] = {"chunk": chunk, "rrf_score": 0.0}
        scores[idx]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        idx = chunk["idx"]
        if idx not in scores:
            scores[idx] = {"chunk": chunk, "rrf_score": 0.0}
        scores[idx]["rrf_score"] += 1.0 / (k + rank + 1)

    fused = sorted(
        scores.values(),
        key     = lambda x: x["rrf_score"],
        reverse = True
    )

    results = []

    for item in fused:
        chunk              = item["chunk"].copy()
        chunk["rrf_score"] = item["rrf_score"]
        results.append(chunk)

    return results


# ==================================================
# QUERY EXPANDER
# (Expands patient question with medical terms)
# ==================================================

def expand_medical_query(query: str) -> str:

    expansions = {
        "cancer"      : "cancer tumor malignancy oncology",
        "lung"        : "lung pulmonary respiratory",
        "breast"      : "breast mammary",
        "survive"     : "survival prognosis survival rate",
        "survive"     : "survival prognosis outcome",
        "treatment"   : "treatment therapy chemotherapy radiation surgery",
        "symptom"     : "symptom sign presentation diagnosis",
        "pain"        : "pain discomfort ache",
        "stage"       : "stage grade classification TNM",
        "chemo"       : "chemotherapy cytotoxic drug treatment",
        "spread"      : "metastasis metastatic spread",
        "biopsy"      : "biopsy pathology histology",
        "cure"        : "cure remission treatment recovery",
        "die"         : "mortality death prognosis survival",
        "tired"       : "fatigue weakness tiredness",
        "blood"       : "blood hematology CBC complete blood count",
    }

    expanded = query

    for word, expansion in expansions.items():
        if word.lower() in query.lower():
            expanded = expanded + " " + expansion

    return expanded


# ==================================================
# HYBRID RETRIEVER CLASS
# ==================================================

class HybridRetriever:

    def __init__(self):

        print("\nInitializing Hybrid Retriever...")

        self.index, self.chunks = load_vector_database()

        print(f"Loading model  : {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Building BM25 index...")
        self.bm25 = build_bm25_index(self.chunks)

        print("Hybrid Retriever ready!\n")


    # ==================================================
    # MAIN RETRIEVE
    # ==================================================

    def retrieve(
        self,
        query            : str,
        top_k            : int  = TOP_K_FINAL,
        expand_query     : bool = True,
        conversation_ctx : str  = ""
    ) -> list:

        # Step 1 — Expand query with medical terms
        if expand_query:
            expanded_query = expand_medical_query(query)
        else:
            expanded_query = query

        # Step 2 — Add conversation context if available
        if conversation_ctx:
            full_query = f"{conversation_ctx} {expanded_query}"
        else:
            full_query = expanded_query

        # Step 3 — Dense retrieval
        dense_results = dense_retrieve(
            query  = full_query,
            index  = self.index,
            chunks = self.chunks,
            model  = self.model,
            top_k  = TOP_K_DENSE
        )

        # Step 4 — Sparse retrieval
        sparse_results = sparse_retrieve(
            query  = full_query,
            bm25   = self.bm25,
            chunks = self.chunks,
            top_k  = TOP_K_BM25
        )

        # Step 5 — Fuse with RRF
        fused_results = reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results
        )

        return fused_results[:top_k]


    # ==================================================
    # FORMAT CONTEXT FOR LLM
    # ==================================================

    def format_context(self, results: list) -> str:

        context_parts = []

        for i, chunk in enumerate(results):

            context_parts.append(
                f"[Medical Source {i+1}]\n"
                f"Document  : {chunk.get('source', 'Unknown')}\n"
                f"Topic     : {chunk.get('title',  'No Title')}\n"
                f"Relevance : {chunk.get('rrf_score', 0):.4f}\n\n"
                f"{chunk['text']}"
            )

        return "\n\n" + "-"*60 + "\n\n".join(context_parts)


# ==================================================
# MAIN — Test Doctor-Patient Retrieval
# ==================================================

if __name__ == "__main__":

    retriever = HybridRetriever()

    # Simulate doctor-patient questions
    patient_questions = [
        "Doctor, I have been coughing for months. Could it be lung cancer?",
        "What are the side effects of chemotherapy?",
        "What is the survival rate for stage 3 breast cancer?",
        "Is cancer hereditary? My mother had breast cancer.",
        "What foods should I avoid during cancer treatment?",
    ]

    for question in patient_questions:

        print(f"\n{'='*60}")
        print(f"Patient : {question}")
        print(f"{'='*60}")

        # Retrieve with query expansion
        results = retriever.retrieve(
            query        = question,
            top_k        = 3,
            expand_query = True
        )

        print(f"\nRetrieved {len(results)} relevant medical chunks:\n")

        for i, chunk in enumerate(results):
            print(f"[{i+1}] Source    : {chunk.get('source', 'Unknown')}")
            print(f"     Topic     : {chunk.get('title', 'No Title')[:60]}")
            print(f"     RRF Score : {chunk.get('rrf_score', 0):.4f}")
            print(f"     Preview   : {chunk['text'][:200]}...")
            print()

        # Show formatted context
        print("Formatted Context for LLM:")
        print(retriever.format_context(results[:2]))