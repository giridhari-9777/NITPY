# src/reranker/cross_encoder.py

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==================================================
# CONFIG
# ==================================================

TOP_K_RERANKED  = 5
SCORE_THRESHOLD = 0.3


# ==================================================
# MEDICAL RERANKER CLASS
# ==================================================

class MedicalReranker:

    def __init__(self):

        print("\nInitializing Medical Reranker...")

        from sentence_transformers import SentenceTransformer, util

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.util = util

        print("Medical Reranker ready!\n")


    # ==================================================
    # RERANK CHUNKS
    # ==================================================

    def rerank(
        self,
        query  : str,
        chunks : list,
        top_k  : int = TOP_K_RERANKED
    ) -> list:

        if not chunks:
            return []

        query_emb = self.model.encode(
            query,
            convert_to_tensor    = True,
            normalize_embeddings = True
        )

        scored_chunks = []

        for chunk in chunks:

            chunk_emb = self.model.encode(
                chunk["text"][:400],
                convert_to_tensor    = True,
                normalize_embeddings = True
            )

            score = self.util.cos_sim(
                query_emb,
                chunk_emb
            ).item()

            chunk                 = chunk.copy()
            chunk["rerank_score"] = float(score)

            scored_chunks.append(chunk)

        scored_chunks.sort(
            key     = lambda x: x["rerank_score"],
            reverse = True
        )

        filtered = [
            chunk for chunk in scored_chunks
            if chunk["rerank_score"] >= SCORE_THRESHOLD
        ]

        if not filtered:
            filtered = scored_chunks[:1]

        return filtered[:top_k]


    # ==================================================
    # FORMAT CONTEXT FOR LLM
    # ==================================================

    def format_context(self, results: list) -> str:

        context_parts = []

        for i, chunk in enumerate(results):

            context_parts.append(
                f"[Medical Reference {i+1}]\n"
                f"Source  : {chunk.get('source', 'Unknown')}\n"
                f"Topic   : {chunk.get('title',  'No Title')}\n"
                f"Score   : {chunk.get('rerank_score', 0):.4f}\n\n"
                f"{chunk['text']}"
            )

        return "\n\n" + "="*60 + "\n\n".join(context_parts)


    # ==================================================
    # RETRIEVE + RERANK PIPELINE
    # ==================================================

    def retrieve_and_rerank(
        self,
        query     : str,
        retriever,
        top_k     : int = TOP_K_RERANKED
    ) -> dict:

        # Step 1 — Hybrid retrieval
        candidates = retriever.retrieve(
            query        = query,
            top_k        = 10,
            expand_query = True
        )

        print(f"  Candidates   : {len(candidates)}")

        # Step 2 — Rerank
        reranked = self.rerank(
            query  = query,
            chunks = candidates,
            top_k  = top_k
        )

        print(f"  After rerank : {len(reranked)}")

        # Step 3 — Format context for generator
        context = self.format_context(reranked)

        return {
            "query"      : query,
            "chunks"     : reranked,
            "context"    : context,
            "num_chunks" : len(reranked)
        }


# ==================================================
# MAIN — Test
# ==================================================

if __name__ == "__main__":

    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    reranker  = MedicalReranker()

    patient_questions = [
        "What are the early signs of lung cancer?",
        "What are the side effects of chemotherapy?",
        "What is the survival rate for stage 2 breast cancer?",
    ]

    for question in patient_questions:

        print(f"\n{'='*60}")
        print(f"Patient  : {question}")
        print(f"{'='*60}")

        result = reranker.retrieve_and_rerank(
            query     = question,
            retriever = retriever,
            top_k     = 3
        )

        for i, chunk in enumerate(result["chunks"]):
            print(f"\n[{i+1}] Source       : {chunk.get('source', 'Unknown')}")
            print(f"     Topic        : {chunk.get('title',  'No Title')[:60]}")
            print(f"     Rerank Score : {chunk.get('rerank_score', 0):.4f}")
            print(f"     Preview      : {chunk['text'][:200]}...")

        print(f"\nContext ready for Generator ✅")
        print(f"Total chunks passed : {result['num_chunks']}")