# src/agents/query_agent.py
# ============================================================
# Query Agent — Groq API version (Render Cloud compatible)
# Features:
#   1. Contextual Memory Layer: Tracks active cancer & resolves "this", "it", "that"
#   2. Strict Doctor Response Rule: Either 1-Line Answer OR Question to Patient (Not Both)
#   3. Bi-directional Doctor-Patient Clinical Simulation
# ============================================================

import os
import sys
import re
import json
import numpy as np
from datetime import datetime

# Add parent paths
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

# Import Memory Manager
try:
    from memory.memory import MemoryManager, KNOWN_CANCERS
except ImportError:
    from src.memory.memory import MemoryManager, KNOWN_CANCERS

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
    """If chroma_db is missing or empty, stream-extract from chroma_db_archive with zero memory overhead."""
    chroma_path = os.path.join(ROOT, "chroma_db")
    sqlite_path = os.path.join(chroma_path, "chroma.sqlite3")

    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 10 * 1024 * 1024:
        return

    archive_dir = os.path.join(ROOT, "chroma_db_archive")
    if not os.path.exists(archive_dir):
        return

    import glob, tarfile
    part_files = sorted(glob.glob(os.path.join(archive_dir, "chroma_db.tar.gz.part_*")))
    if not part_files:
        return

    print(f"Extracting ChromaDB from {len(part_files)} archive parts (zero-memory stream)...")
    temp_archive = os.path.join(ROOT, "temp_chroma_extract.tar.gz")
    try:
        with open(temp_archive, "wb") as out_f:
            for p in part_files:
                with open(p, "rb") as in_f:
                    while True:
                        chunk = in_f.read(65536)
                        if not chunk:
                            break
                        out_f.write(chunk)

        with tarfile.open(temp_archive, "r:gz") as tar:
            tar.extractall(path=ROOT)
        print("ChromaDB extracted successfully!")
    except Exception as e:
        print(f"ChromaDB extraction error: {e}")
    finally:
        if os.path.exists(temp_archive):
            try:
                os.remove(temp_archive)
            except Exception:
                pass


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
    "symptoms"    : ["symptom","sign","feel","hurt","pain","discomfort","notice","cough","blood","fatigue","lump"],
    "diagnosis"   : ["diagnose","diagnosis","test","biopsy","scan","detect","check","mri","ct","mammogram","psa"],
    "treatment"   : ["treat","therapy","chemo","radiation","surgery","medication","drug","regimen","folfox","immunotherapy"],
    "prognosis"   : ["survive","survival","prognosis","outlook","life","stage","chance","remission"],
    "prevention"  : ["prevent","reduce","risk","avoid","protect","screening","diet","lifestyle"],
    "side_effects": ["side effect","nausea","fatigue","hair loss","vomit","effect","neuropathy","toxicity"],
    "staging"     : ["stage","staging","spread","metasta","grade","tnm","node"],
    "mechanism"   : ["cause","why","how","mechanism","develop","form","growth","mutation","gene"],
    "history"     : ["history","smoking","smoke","family","prior","exposure","pack"],
    "general"     : ["what is","define","explain","about","information","overview","hello","hi"],
}


# ============================================================
# QUERY AGENT CLASS
# ============================================================

class QueryAgent:
    """
    Oncology QA Agent using Groq API + ChromaDB RAG.
    Integrated with persistent Memory Layer and strict response formatting.
    """

    def __init__(self):
        self.memory_manager = MemoryManager()
        print("QueryAgent initialized (Memory Layer & One-Line/Question Rule active)")

    # ── Process question ──────────────────────────────────────

    def process(
        self,
        question              : str,
        session_id            : str  = "default",
        mode                  : str  = "patient",
        patient_profile       : dict = None,
        doctor_response_style : str  = "auto"  # 'auto', 'answer_only', 'question_only'
    ) -> dict:
        """
        Main entry point for processing patient or doctor inquiries.
        """

        # 1. Non-medical rejection check
        if self._is_non_medical(question, mode=mode):
            return self._reject_response(question)

        # 2. Memory Layer: Resolve pronouns ("this", "it", "that", "this cancer")
        resolved_question, was_resolved = self.memory_manager.resolve(session_id, question)

        # 3. Detect / update cancer type
        if mode == "doctor" and patient_profile:
            cancer = patient_profile.get("cancer_type", "lung cancer")
        else:
            detected_cancer = self._detect_cancer(resolved_question)
            if detected_cancer:
                cancer = detected_cancer
            else:
                cancer = self.memory_manager.get_cancer_context(session_id)

        # 4. Detect question type
        qtype = self._detect_type(resolved_question)

        # 5. Retrieve from ChromaDB
        search_query = f"{cancer} {resolved_question}" if cancer else resolved_question
        chunks, context = self._retrieve(search_query)

        # 6. Generate appropriate response
        if mode == "doctor":
            # AI is the Simulated Patient
            answer = self._generate_patient_response(
                question   = question,
                context    = context,
                profile    = patient_profile or {},
                session_id = session_id
            )
            action = "patient_reply"

        elif mode == "doctor_qa":
            # AI is Clinical Specialist Copilot
            answer = self._generate_clinical_response(
                question   = resolved_question,
                context    = context,
                cancer     = cancer,
                qtype      = qtype,
                session_id = session_id
            )
            action = "clinical_answer"

        else:
            # Mode = Patient (AI is Doctor)
            # Rule: Either 1-Line Answer OR Clinical Question to Patient (Not Both)
            answer, action = self._generate_doctor_response(
                question       = resolved_question,
                context        = context,
                cancer         = cancer,
                qtype          = qtype,
                session_id     = session_id,
                response_style = doctor_response_style
            )

        # 7. Update Memory Layer
        self.memory_manager.add_turn(
            session_id    = session_id,
            question      = resolved_question,
            answer        = answer,
            cancer_type   = cancer,
            question_type = qtype,
            action        = action,
            role          = "doctor" if mode == "doctor" else "patient"
        )

        # 8. Score & hallucination check
        confidence = self._score_answer(answer, context, chunks)
        hall       = self._hallucination_check(answer, context, chunks, mode=mode)

        # 9. Build sources list
        sources = [
            {
                "source" : c.get("source", "unknown"),
                "score"  : round(c.get("rerank_score", 0), 3)
            }
            for c in chunks[:3]
        ]

        return {
            "answer"             : answer,
            "question"           : question,
            "resolved_question"  : resolved_question,
            "was_resolved"       : was_resolved,
            "action"             : action,              # 'answer' or 'question'
            "question_type"      : qtype,
            "cancer_type"        : cancer,
            "confidence"         : confidence,
            "sources"            : sources,
            "hallucination"      : hall,
            "memory_context"     : self.memory_manager.get_context_str(session_id),
            "chunks_used"        : len(chunks),
            "mode"               : mode,
        }

    # ── Doctor AI Response Generation (Strict 1-Line OR Question) ─

    def _generate_doctor_response(
        self,
        question       : str,
        context        : str,
        cancer         : str,
        qtype          : str,
        session_id     : str,
        response_style : str = "auto"
    ) -> tuple:
        """
        Generates either:
          - A concise 1-line answer (like a QA dataset record), OR
          - A relevant clinical follow-up question to the patient.
        STRICT RULE: Never both.
        Returns: (response_text: str, action: str)
        """

        # Decide action
        q_low = question.lower()
        personal_symptom_indicators = [
            "i feel", "i have", "i'm having", "my doctor said", "i noticed",
            "i found", "i cough", "i bleed", "scared", "worried", "hurts me"
        ]

        if response_style == "answer_only":
            action = "answer"
        elif response_style == "question_only":
            action = "question"
        else:
            # Auto decide:
            # If patient is reporting personal symptoms/feelings -> Doctor asks a clinical question
            # If patient is asking a factual question (what, how, why, survival, treatment) -> Doctor answers in 1 line
            if any(ind in q_low for ind in personal_symptom_indicators):
                action = "question"
            else:
                action = "answer"

        mem_str = self.memory_manager.get_context_str(session_id, last_n=3)

        if action == "answer":
            # ── 1-LINE DIRECT ANSWER ─────────────────────────
            system = (
                "You are an expert oncologist AI. Provide an accurate, direct ONE-LINE answer "
                "to the patient's cancer question, formatted like a clean QA JSON dataset entry. "
                "STRICT INSTRUCTION: Provide exactly ONE clear, concise medical sentence. "
                "DO NOT ask any question. Do not include follow-up questions."
            )

            prompt = f"""PATIENT QUESTION: {question}
{f'CANCER TOPIC: {cancer}' if cancer else ''}

RETRIEVED ONCOLOGY TEXTBOOK CONTEXT:
{context[:1500] if context else 'Standard oncology medical knowledge.'}

INSTRUCTION: Write exactly ONE concise sentence answering the question directly. No questions allowed.
ONE-LINE ANSWER:"""

            raw_resp = call_groq(
                prompt      = prompt,
                model       = "llama3",
                max_tokens  = 120,
                temperature = 0.1,
                system      = system
            )

            if not raw_resp:
                raw_resp = f"Treatment and management for {cancer or 'cancer'} depend on the tumor stage, histology, and patient overall health."

            # Clean and ensure strict 1-line answer with no questions
            answer = self._clean_one_line_answer(raw_resp)
            return answer, "answer"

        else:
            # ── RELEVANT CLINICAL QUESTION TO PATIENT ─────────
            system = (
                "You are an empathetic oncologist consulting with a patient. "
                "Ask the patient exactly ONE relevant, compassionate clinical follow-up question "
                "to investigate their symptoms, timeline, severity, or medical history. "
                "STRICT INSTRUCTION: Output ONLY the question ending with a question mark. "
                "DO NOT give an answer or medical explanation."
            )

            prompt = f"""PATIENT STATEMENT / INQUIRY: {question}
{f'ACTIVE CANCER CONTEXT: {cancer}' if cancer else ''}
{f'PRIOR TURNS: {mem_str}' if mem_str else ''}

INSTRUCTION: Ask the patient exactly ONE clinical follow-up question related to what they said. Do NOT give an answer.
DOCTOR'S QUESTION TO PATIENT:"""

            raw_resp = call_groq(
                prompt      = prompt,
                model       = "llama3",
                max_tokens  = 80,
                temperature = 0.3,
                system      = system
            )

            if not raw_resp:
                raw_resp = f"How long have you been experiencing these symptoms, and have you discussed them with your primary doctor?"

            # Clean and ensure strict question format
            question_text = self._clean_question_only(raw_resp)
            return question_text, "question"

    # ── Cleaners to enforce "Not Both" ────────────────────────

    def _clean_one_line_answer(self, text: str) -> str:
        """Strips thinking tokens, newlines, and trailing questions."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        first_chunk = lines[0] if lines else text

        # Split sentences and discard any sentence ending with ?
        sentences = re.split(r"(?<=[.!?])\s+", first_chunk)
        clean_sentences = [s.strip() for s in sentences if not s.strip().endswith("?")]

        if clean_sentences:
            res = clean_sentences[0]
        else:
            res = sentences[0].rstrip("?") + "."

        return res.strip()

    def _clean_question_only(self, text: str) -> str:
        """Ensures the text is solely a single question ending with ?."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        joined = " ".join(lines)

        questions = re.findall(r"([^.!?]*\?)", joined)
        if questions:
            return questions[0].strip()
        else:
            return joined.strip().rstrip(".") + "?"

    # ── Patient AI Simulation (When User is Doctor) ───────────

    def _generate_patient_response(
        self,
        question   : str,
        context    : str,
        profile    : dict,
        session_id : str
    ) -> str:
        """Generate in-character response as a simulated cancer patient."""

        if not profile:
            profile = {
                "name"            : "Robert Miller",
                "age"             : 58,
                "gender"          : "Male",
                "cancer_type"     : "lung cancer",
                "stage"           : "Suspected NSCLC",
                "chief_complaint" : "Persistent dry cough for 3 months, fatigue, rust-colored sputum.",
                "medical_history" : "35 pack-year smoker.",
                "emotional_state" : "Anxious, seeking clarity.",
            }

        mem_str = self.memory_manager.get_context_str(session_id, last_n=3)

        system = (
            f"You are roleplaying as a simulated cancer patient named {profile.get('name', 'Patient')}, "
            f"a {profile.get('age', 55)}-year-old {profile.get('gender', 'person')} in a clinical interview with your examining doctor.\n"
            f"YOUR PROFILE:\n"
            f"• Condition: {profile.get('cancer_type', 'cancer')} ({profile.get('stage', 'Under Evaluation')})\n"
            f"• Chief Complaint: {profile.get('chief_complaint', 'Symptoms')}\n"
            f"• History: {profile.get('medical_history', 'None')}\n"
            f"• Tone: {profile.get('emotional_state', 'Anxious')}\n\n"
            f"RULES:\n"
            f"1. Speak strictly in first person as the patient ('Doctor, I feel...').\n"
            f"2. Never say you are an AI.\n"
            f"3. Answer the doctor's specific inquiry in 2-3 natural sentences."
        )

        prompt = f"""DOCTOR'S QUESTION: {question}
{f'CONVERSATION MEMORY: {mem_str}' if mem_str else ''}
{f'ONCOLOGY CONTEXT: {context[:1000]}' if context else ''}

PATIENT RESPONSE:"""

        answer = call_groq(
            prompt      = prompt,
            model       = "llama3",
            max_tokens  = 250,
            temperature = 0.4,
            system      = system
        )

        if not answer:
            answer = f"Doctor, I've had {profile.get('chief_complaint', 'these symptoms')} and I'm really hoping you can help me."

        return answer

    # ── Doctor QA / Clinical Specialist Mode ──────────────────

    def _generate_clinical_response(
        self,
        question   : str,
        context    : str,
        cancer     : str,
        qtype      : str,
        session_id : str
    ) -> str:
        """Provide specialist-level oncological reference for practicing physicians."""

        mem_str = self.memory_manager.get_context_str(session_id, last_n=3)

        system = (
            "You are an expert Oncology Clinical Copilot. Provide rigorous, evidence-based "
            "clinical guidance utilizing NCCN/ESMO guidelines, TNM staging, molecular biomarker targets, "
            "and systemic regimens with clinical precision."
        )

        prompt = f"""PHYSICIAN CLINICAL INQUIRY: {question}
{f'CANCER TYPE: {cancer}' if cancer else ''}

RETRIEVED ONCOLOGY CONTEXT:
{context[:1800] if context else 'Standard oncology clinical guidelines apply.'}

CLINICAL GUIDANCE:"""

        answer = call_groq(
            prompt      = prompt,
            model       = "llama3",
            max_tokens  = 400,
            temperature = 0.1,
            system      = system
        )

        if not answer:
            answer = "Clinical guidance unavailable. Please consult current NCCN/ASCO oncology guidelines."

        return answer

    # ── Cancer Detection ──────────────────────────────────────

    def _detect_cancer(self, question: str) -> str:
        q_low = question.lower()
        for cancer in KNOWN_CANCERS:
            if cancer in q_low:
                return cancer
        return ""

    def _detect_type(self, question: str) -> str:
        q = question.lower()
        for qtype, keywords in QUESTION_TYPES.items():
            if any(kw in q for kw in keywords):
                return qtype
        return "general"

    def _is_non_medical(self, question: str, mode: str = "patient") -> bool:
        if mode in ["doctor", "doctor_qa"]:
            return False

        q = question.lower()
        clinical_allow = [
            "how are you", "how do you feel", "what brings you", "symptom",
            "cough", "pain", "lump", "bleed", "breath", "smok", "weight",
            "test", "scan", "biopsy", "history", "feel", "fever", "doctor",
            "this", "it", "treatment", "stage", "cure", "survive", "chemo"
        ]
        if any(term in q for term in clinical_allow):
            return False

        for pat in NON_MEDICAL:
            if re.search(pat, q, re.IGNORECASE):
                return True
        return False

    def _reject_response(self, question: str) -> dict:
        return {
            "answer"            : "I am an oncology medical AI assistant. Please ask questions related to cancer symptoms, diagnosis, treatment, or simulation.",
            "resolved_question" : question,
            "was_resolved"      : False,
            "action"            : "reject",
            "question_type"     : "non_medical",
            "cancer_type"       : "",
            "confidence"        : 0.0,
            "sources"           : [],
            "hallucination"     : {"score": 0.0, "verdict": "", "safety": "LOW"},
            "memory_context"    : "",
            "chunks_used"       : 0,
            "mode"              : "patient"
        }

    # ── ChromaDB Retrieval ────────────────────────────────────

    def _retrieve(self, question: str, k: int = 5) -> tuple:
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
                include          = ["documents", "metadatas", "distances"]
            )

            chunks = []
            for i in range(len(result["ids"][0])):
                raw   = 1 - result["distances"][0][i]
                text  = result["documents"][0][i]
                src   = result["metadatas"][0][i].get("source", "unknown")

                c_emb = emb_model.encode(
                    text[:500],
                    normalize_embeddings = True,
                    convert_to_numpy     = True
                )
                rerank = float(np.dot(q_emb, c_emb))

                chunks.append({
                    "text"         : text,
                    "source"       : src,
                    "raw_score"    : round(raw, 4),
                    "rerank_score" : round(rerank, 4),
                })

            chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            context = "\n\n---\n\n".join([c["text"] for c in chunks[:k]])
            return chunks, context

        except Exception as e:
            print(f"Retrieval error: {e}")
            return [], ""

    # ── Scoring & Hallucination Check ─────────────────────────

    def _score_answer(self, answer: str, context: str, chunks: list) -> float:
        if not answer or not chunks:
            return 3.5
        emb_model = _load_emb_model()
        if emb_model is None:
            return 3.5
        try:
            a_emb = emb_model.encode(answer[:500], normalize_embeddings=True, convert_to_numpy=True)
            sims = []
            for c in chunks[:5]:
                c_emb = emb_model.encode(c["text"][:500], normalize_embeddings=True, convert_to_numpy=True)
                sims.append(float(np.dot(a_emb, c_emb)))
            if not sims:
                return 3.5
            score = round(1.0 + float(np.mean(sims)) * 4.0, 2)
            return min(5.0, max(1.0, score))
        except Exception:
            return 3.5

    def _hallucination_check(self, answer: str, context: str, chunks: list, mode: str = "patient") -> dict:
        if not answer:
            return {"score": 0.0, "verdict": "EMPTY", "safety": "HIGH", "is_hallucinated": True, "faithfulness": 0.0, "relevance": 0.0}

        unsafe_patterns = [
            r"100\s*%\s*(cure|cured|effective)",
            r"guaranteed\s+(cure|recovery)",
            r"definitely\s+(cures|treats)",
            r"no\s+side\s+effects",
            r"miracle\s+(cure|treatment)",
        ]
        is_unsafe = any(re.search(p, answer.lower()) for p in unsafe_patterns)
        faith = 0.75
        rel   = 0.75

        emb_model = _load_emb_model()
        if emb_model and chunks:
            try:
                a_emb = emb_model.encode(answer[:400], normalize_embeddings=True, convert_to_numpy=True)
                sims = []
                for c in chunks[:3]:
                    c_emb = emb_model.encode(c["text"][:400], normalize_embeddings=True, convert_to_numpy=True)
                    sims.append(float(np.dot(a_emb, c_emb)))
                if sims:
                    faith = round(float(np.mean(sims)), 4)
                    rel   = round(float(max(sims)), 4)
            except Exception:
                pass

        score   = round(faith * 0.7 + (0 if is_unsafe else 0.3), 2)
        score_5 = min(5.0, max(1.0, round(1.0 + score * 4.0, 2)))
        is_hall = (faith < 0.30 and mode != "doctor") or is_unsafe

        verdict = "❌ UNSAFE CLAIM" if is_unsafe else ("⚠️ LOW FAITHFULNESS" if faith < 0.30 else "✅ PASS")
        safety  = "HIGH" if is_unsafe else ("MEDIUM" if faith < 0.45 else "LOW")

        return {
            "score"          : score_5,
            "verdict"        : verdict,
            "safety"         : safety,
            "is_hallucinated": bool(is_hall),
            "faithfulness"   : faith,
            "relevance"      : rel,
        }

    def clear_memory(self, session_id: str):
        self.memory_manager.clear_session(session_id)
