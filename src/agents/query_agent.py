# src/agents/query_agent.py
# ============================================================
# Query Agent — Groq API version (Render Cloud compatible)
# Replaces Ollama with Groq API calls
# Same interface as original — ui.py needs no changes
# ============================================================

import os
import sys
import re
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

# Add parent paths
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

# Import Groq client
try:
    from agents.groq_client import call_groq
except ImportError:
    from src.agents.groq_client import call_groq

# ── Lazy imports ──────────────────────────────────────────────
_chroma     = None
_collection = None
_emb_model  = None



def _ensure_chroma_extracted():
    """If chroma_db is missing or empty, automatically extract from chroma_db_archive parts."""
    chroma_path = os.path.join(ROOT, "chroma_db")
    sqlite_path = os.path.join(chroma_path, "chroma.sqlite3")

    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 10 * 1024 * 1024:
        return

    archive_dir = os.path.join(ROOT, "chroma_db_archive")
    if not os.path.exists(archive_dir):
        return

    import glob, tarfile, io
    part_files = sorted(glob.glob(os.path.join(archive_dir, "chroma_db.tar.gz.part_*")))
    if not part_files:
        return

    print(f"Extracting ChromaDB ({len(part_files)} parts) into {ROOT}...")
    try:
        combined_bytes = bytearray()
        for p in part_files:
            with open(p, "rb") as f:
                combined_bytes.extend(f.read())

        bio = io.BytesIO(combined_bytes)
        with tarfile.open(fileobj=bio, mode="r:gz") as tar:
            tar.extractall(path=ROOT)
        print("ChromaDB extracted successfully!")
    except Exception as e:
        print(f"ChromaDB extraction error: {e}")


def _load_chroma():
    """Load ChromaDB (lazy — only once)."""
    global _chroma, _collection
    if _collection is not None:
        return _collection

    _ensure_chroma_extracted()

    import chromadb
    try:
        _chroma = chromadb.PersistentClient(
            path = os.path.join(ROOT, "chroma_db")
        )
        _collection = _chroma.get_or_create_collection(
            name     = "medical_rag",
            metadata = {"hnsw:space": "cosine"}
        )
        print(f"ChromaDB loaded: {_collection.count()} records")
    except Exception as e:
        print(f"ChromaDB error: {e}")
        _collection = None

    return _collection


def _load_emb_model():
    """Load sentence transformer (lazy — only once)."""
    global _emb_model
    if _emb_model is not None:
        return _emb_model

    from sentence_transformers import SentenceTransformer
    try:
        _emb_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model loaded")
    except Exception as e:
        print(f"Embedding model error: {e}")
        _emb_model = None

    return _emb_model


# ── Cancer types ──────────────────────────────────────────────
CANCER_TYPES = [
    "lung cancer", "breast cancer", "colon cancer",
    "colorectal cancer", "prostate cancer", "skin cancer",
    "melanoma", "leukemia", "lymphoma", "ovarian cancer",
    "cervical cancer", "pancreatic cancer", "liver cancer",
    "stomach cancer", "kidney cancer", "bladder cancer",
    "thyroid cancer", "brain tumor", "bone cancer",
    "esophageal cancer", "testicular cancer", "uterine cancer",
]

# ── Non-medical patterns ──────────────────────────────────────
NON_MEDICAL = [
    r"\b(weather|temperature|rain|sunny|climate)\b",
    r"\b(stock|market|bitcoin|crypto|price)\b",
    r"\b(movie|film|song|music|celebrity|actor)\b",
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(sports|football|cricket|ipl|nba)\b",
    r"\b(code|program|python|javascript|bug|error)\b",
    r"\b(politics|election|government|president)\b",
]

# ── Question types ────────────────────────────────────────────
QUESTION_TYPES = {
    "symptoms"    : ["symptom","sign","feel","hurt","pain","discomfort","notice"],
    "diagnosis"   : ["diagnose","diagnosis","test","biopsy","scan","detect","check"],
    "treatment"   : ["treat","therapy","chemo","radiation","surgery","medication","drug"],
    "prognosis"   : ["survive","survival","prognosis","outlook","life","stage","chance"],
    "prevention"  : ["prevent","reduce","risk","avoid","protect","screening"],
    "side_effects": ["side effect","nausea","fatigue","hair loss","vomit","effect"],
    "staging"     : ["stage","staging","spread","metasta","grade"],
    "mechanism"   : ["cause","why","how","mechanism","develop","form","growth"],
    "general"     : ["what is","define","explain","about","information","overview"],
}


# ============================================================
# QUERY AGENT CLASS
# ============================================================

class QueryAgent:
    """
    Oncology QA Agent using Groq API + ChromaDB RAG.
    Drop-in replacement for Ollama-based agent.
    Same .process() interface — ui.py unchanged.
    """

    def __init__(self):
        self.memory       = defaultdict(list)
        self.cancer_memory= {}
        print("QueryAgent initialized (Groq API mode)")

    # ── Process question ──────────────────────────────────────

    def process(
        self,
        question   : str,
        session_id : str = "default"
    ) -> dict:
        """
        Main entry point — same interface as original.
        Returns dict with answer, metadata, scores.
        """

        # 1. Check if medical question
        if self._is_non_medical(question):
            return self._reject_response(question)

        # 2. Detect question type and cancer type
        qtype  = self._detect_type(question)
        cancer = self._detect_cancer(question, session_id)

        # 3. Check if follow-up question
        is_followup = self._is_followup(question)

        # 4. Resolve pronouns using memory
        resolved = self._resolve_question(
            question, session_id, cancer
        )

        # 5. Retrieve from ChromaDB
        chunks, context = self._retrieve(resolved)

        # 6. Generate answer via Groq
        answer = self._generate(
            question = resolved,
            context  = context,
            cancer   = cancer,
            qtype    = qtype,
            session_id = session_id
        )

        # 7. Score answer
        confidence = self._score_answer(
            answer, context, chunks
        )

        # 8. Hallucination check
        hall = self._hallucination_check(
            answer, context, chunks
        )

        # 9. Save to memory
        self._save_memory(
            session_id, question, answer, cancer
        )

        # 10. Build sources list
        sources = [
            {
                "source" : c.get("source", "unknown"),
                "score"  : round(c.get("rerank_score", 0), 3)
            }
            for c in chunks[:3]
        ]

        return {
            "answer"           : answer,
            "question"         : question,
            "resolved_question": resolved,
            "question_type"    : qtype,
            "cancer_type"      : cancer,
            "is_followup"      : is_followup,
            "confidence"       : confidence,
            "sources"          : sources,
            "hallucination"    : hall,
            "memory_context"   : self._get_memory_str(session_id),
            "chunks_used"      : len(chunks),
        }

    # ── Non-medical check ─────────────────────────────────────

    def _is_non_medical(self, question: str) -> bool:
        q = question.lower()
        for pat in NON_MEDICAL:
            if re.search(pat, q, re.IGNORECASE):
                return True
        return False

    def _reject_response(self, question: str) -> dict:
        return {
            "answer"        : (
                "I'm a specialized oncology assistant and can only "
                "answer cancer-related medical questions. "
                "Please ask me about cancer symptoms, diagnosis, "
                "treatment, or prevention."
            ),
            "question_type" : "non_medical",
            "cancer_type"   : "",
            "is_followup"   : False,
            "confidence"    : 0.0,
            "sources"       : [],
            "hallucination" : {"score": 0.0, "verdict": "", "safety": "LOW"},
        }

    # ── Detect question type ──────────────────────────────────

    def _detect_type(self, question: str) -> str:
        q = question.lower()
        for qtype, keywords in QUESTION_TYPES.items():
            if any(kw in q for kw in keywords):
                return qtype
        return "general"

    # ── Detect cancer type ────────────────────────────────────

    def _detect_cancer(
        self, question: str, session_id: str
    ) -> str:
        q = question.lower()
        for cancer in CANCER_TYPES:
            if cancer in q:
                self.cancer_memory[session_id] = cancer
                return cancer

        # Return from memory if follow-up
        return self.cancer_memory.get(session_id, "")

    # ── Follow-up detection ───────────────────────────────────

    def _is_followup(self, question: str) -> bool:
        q = question.lower().strip()
        followup_indicators = [
            q.startswith("what about"),
            q.startswith("and "),
            q.startswith("also "),
            q.startswith("what is the"),
            "this cancer" in q,
            "the cancer" in q,
            q.startswith("how about"),
            len(q.split()) <= 5,
        ]
        return any(followup_indicators)

    # ── Resolve pronouns ──────────────────────────────────────

    def _resolve_question(
        self,
        question   : str,
        session_id : str,
        cancer     : str
    ) -> str:
        if not cancer:
            return question

        q = question
        for pronoun in ["this cancer", "the cancer", "it"]:
            if pronoun in q.lower():
                q = re.sub(
                    pronoun, cancer,
                    q, flags=re.IGNORECASE
                )
        return q

    # ── ChromaDB Retrieval ────────────────────────────────────

    def _retrieve(
        self, question: str, k: int = 5
    ) -> tuple:

        collection = _load_chroma()
        emb_model  = _load_emb_model()

        if collection is None or emb_model is None:
            return [], ""

        try:
            q_emb = emb_model.encode(
                question,
                normalize_embeddings = True,
                convert_to_numpy     = True
            )

            result = collection.query(
                query_embeddings = [q_emb.tolist()],
                n_results        = k,
                include          = [
                    "documents","metadatas","distances"
                ]
            )

            chunks = []
            for i in range(len(result["ids"][0])):
                raw   = 1 - result["distances"][0][i]
                text  = result["documents"][0][i]
                src   = result["metadatas"][0][i].get(
                    "source", "unknown"
                )

                # Re-rank score
                c_emb = emb_model.encode(
                    text[:500],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                rerank = float(np.dot(q_emb, c_emb))

                chunks.append({
                    "text"         : text,
                    "source"       : src,
                    "raw_score"    : round(raw,    4),
                    "rerank_score" : round(rerank, 4),
                })

            # Sort by rerank score
            chunks = sorted(
                chunks,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            context = "\n\n---\n\n".join([
                c["text"] for c in chunks[:k]
            ])

            return chunks, context

        except Exception as e:
            print(f"Retrieval error: {e}")
            return [], ""

    # ── Generate answer via Groq ──────────────────────────────

    def _generate(
        self,
        question   : str,
        context    : str,
        cancer     : str,
        qtype      : str,
        session_id : str
    ) -> str:

        # Build memory context
        mem_str = self._get_memory_str(session_id)

        # Build system prompt
        system = (
            "You are a compassionate and knowledgeable oncologist "
            "AI assistant. You provide accurate, evidence-based "
            "information about cancer. Always use appropriate "
            "medical hedging language (may, typically, research "
            "suggests, consult your doctor). Never make absolute "
            "claims or guarantee outcomes. Be empathetic and clear."
        )

        if context:
            prompt = f"""You are an expert oncologist AI assistant.

RETRIEVED MEDICAL CONTEXT:
{context[:2000]}

{f'CONVERSATION MEMORY: {mem_str}' if mem_str else ''}
{f'CANCER TYPE: {cancer}' if cancer else ''}
QUESTION TYPE: {qtype}

PATIENT QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY based on the provided context
- Use hedging language: "may", "typically", "research suggests"
- Be empathetic and compassionate
- Recommend consulting an oncologist for personal medical advice
- Answer in 3-5 clear sentences
- Do NOT make absolute claims

ANSWER:"""
        else:
            # Fallback if no context retrieved
            prompt = f"""You are an expert oncologist AI assistant.

{f'CONVERSATION MEMORY: {mem_str}' if mem_str else ''}
{f'CANCER TYPE: {cancer}' if cancer else ''}

PATIENT QUESTION: {question}

INSTRUCTIONS:
- Provide helpful general information about cancer
- Use hedging language: "may", "typically", "research suggests"
- Be empathetic and compassionate
- Strongly recommend consulting an oncologist
- Answer in 3-5 clear sentences

ANSWER:"""

        answer = call_groq(
            prompt      = prompt,
            model       = "llama3",
            max_tokens  = 400,
            temperature = 0.1,
            system      = system
        )

        if not answer:
            answer = (
                "I apologize, I couldn't generate a response right now. "
                "Please try again in a moment or consult a qualified oncologist."
            )

        return answer

    # ── Score answer ──────────────────────────────────────────

    def _score_answer(
        self,
        answer : str,
        context: str,
        chunks : list
    ) -> float:

        if not answer or not chunks:
            return 0.0

        emb_model = _load_emb_model()
        if emb_model is None:
            return 3.0

        try:
            a_emb = emb_model.encode(
                answer[:500],
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            sims = []
            for c in chunks[:5]:
                c_emb = emb_model.encode(
                    c["text"][:500],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                sims.append(float(np.dot(a_emb, c_emb)))

            if not sims:
                return 3.0

            avg_sim = float(np.mean(sims))

            # Scale 0-1 cosine sim → 1-5 score
            score = round(1.0 + avg_sim * 4.0, 2)
            return min(5.0, max(1.0, score))

        except Exception:
            return 3.0

    # ── Hallucination check ───────────────────────────────────

    def _hallucination_check(
        self,
        answer : str,
        context: str,
        chunks : list
    ) -> dict:

        if not answer:
            return {
                "score"          : 0.0,
                "verdict"        : "EMPTY",
                "safety"         : "HIGH",
                "is_hallucinated": True,
                "faithfulness"   : 0.0,
                "relevance"      : 0.0,
            }

        # Safety check
        unsafe_patterns = [
            r"100\s*%\s*(cure|cured|effective)",
            r"guaranteed\s+(cure|recovery)",
            r"definitely\s+(cures|treats)",
            r"no\s+side\s+effects",
            r"miracle\s+(cure|treatment)",
        ]

        a_lower   = answer.lower()
        is_unsafe = any(
            re.search(p, a_lower)
            for p in unsafe_patterns
        )

        # Faithfulness check
        emb_model = _load_emb_model()
        faith = 0.7
        rel   = 0.7

        if emb_model and chunks:
            try:
                a_emb = emb_model.encode(
                    answer[:400],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                sims = []
                for c in chunks[:3]:
                    c_emb = emb_model.encode(
                        c["text"][:400],
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    sims.append(float(np.dot(a_emb, c_emb)))
                if sims:
                    faith = round(float(np.mean(sims)), 4)
                    rel   = round(float(max(sims)),      4)
            except Exception:
                pass

        # Overall score (higher = better)
        score = round(faith * 0.7 + (0 if is_unsafe else 0.3), 2)

        # Scale to 1-5
        score_5 = round(1.0 + score * 4.0, 2)
        score_5 = min(5.0, max(1.0, score_5))

        is_hall = faith < 0.40 or is_unsafe

        verdict = (
            "❌ UNSAFE CLAIM"       if is_unsafe  else
            "⚠️ LOW FAITHFULNESS"  if faith < 0.40 else
            "✅ PASS"
        )

        safety  = "HIGH" if is_unsafe else "MEDIUM" if faith < 0.5 else "LOW"

        return {
            "score"          : score_5,
            "verdict"        : verdict,
            "safety"         : safety,
            "is_hallucinated": bool(is_hall),
            "faithfulness"   : faith,
            "relevance"      : rel,
        }

    # ── Memory ────────────────────────────────────────────────

    def _save_memory(
        self,
        session_id : str,
        question   : str,
        answer     : str,
        cancer     : str
    ):
        self.memory[session_id].append({
            "q"      : question[:100],
            "a"      : answer[:200],
            "cancer" : cancer,
            "time"   : datetime.now().isoformat()
        })

        # Keep last 5 turns only
        if len(self.memory[session_id]) > 5:
            self.memory[session_id] = (
                self.memory[session_id][-5:]
            )

    def _get_memory_str(self, session_id: str) -> str:
        mem = self.memory.get(session_id, [])
        if not mem:
            return ""
        lines = []
        for m in mem[-3:]:
            if m.get("cancer"):
                lines.append(f"Cancer: {m['cancer']}")
            lines.append(f"Q: {m['q'][:60]}")
            lines.append(f"A: {m['a'][:100]}")
        return "\n".join(lines)

    def clear_memory(self, session_id: str):
        self.memory.pop(session_id, None)
        self.cancer_memory.pop(session_id, None)
