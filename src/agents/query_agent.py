# src/agents/query_agent.py
# ============================================================
# Query Agent — Groq API version (Render Cloud compatible)
# Supports Bi-directional Doctor-Patient Simulation:
#   1. Patient Mode (User = Patient, AI = Doctor)
#   2. Doctor Mode (User = Doctor, AI = Simulated Patient)
#   3. Clinical Mode (User = Doctor, AI = Oncology Specialist)
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
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
    Supports:
      - mode='patient'    : User is Patient, AI is Doctor (compassionate oncologist)
      - mode='doctor'     : User is Doctor, AI is Patient (simulated realistic patient)
      - mode='doctor_qa'  : User is Doctor, AI is Oncology Specialist Copilot
    """

    def __init__(self):
        self.memory        = defaultdict(list)
        self.cancer_memory = {}
        print("QueryAgent initialized (Groq API mode — Bidirectional Simulation ready)")

    # ── Process question ──────────────────────────────────────

    def process(
        self,
        question        : str,
        session_id      : str  = "default",
        mode            : str  = "patient",
        patient_profile : dict = None
    ) -> dict:
        """
        Main entry point.
        Args:
            question        : Text inquiry
            session_id      : Unique session key for conversation memory
            mode            : 'patient' (User=Patient), 'doctor' (User=Doctor), or 'doctor_qa'
            patient_profile : Dict case profile when mode='doctor'
        """

        # 1. Non-medical rejection check
        if self._is_non_medical(question, mode=mode):
            return self._reject_response(question)

        # 2. Determine cancer type
        if mode == "doctor" and patient_profile:
            cancer = patient_profile.get("cancer_type", "lung cancer")
        else:
            cancer = self._detect_cancer(question, session_id)

        # 3. Detect question type
        qtype = self._detect_type(question)

        # 4. Check if follow-up
        is_followup = self._is_followup(question)

        # 5. Resolve question with memory
        resolved = self._resolve_question(
            question, session_id, cancer
        )

        # 6. Retrieve relevant medical textbook context from ChromaDB
        search_query = f"{cancer} {resolved}" if cancer else resolved
        chunks, context = self._retrieve(search_query)

        # 7. Generate appropriate response based on simulation mode
        if mode == "doctor":
            # AI responds as the Simulated Cancer Patient
            answer = self._generate_patient_response(
                question   = question,
                context    = context,
                profile    = patient_profile or {},
                session_id = session_id
            )
        elif mode == "doctor_qa":
            # AI responds as Clinical Oncology Specialist
            answer = self._generate_clinical_response(
                question   = resolved,
                context    = context,
                cancer     = cancer,
                qtype      = qtype,
                session_id = session_id
            )
        else:
            # Standard Mode: AI responds as compassionate Doctor AI
            answer = self._generate(
                question   = resolved,
                context    = context,
                cancer     = cancer,
                qtype      = qtype,
                session_id = session_id
            )

        # 8. Score answer & hallucination verification
        confidence = self._score_answer(answer, context, chunks)
        hall       = self._hallucination_check(answer, context, chunks, mode=mode)

        # 9. Save memory
        self._save_memory(session_id, question, answer, cancer, mode=mode)

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
            "mode"             : mode,
        }

    # ── Non-medical check ─────────────────────────────────────

    def _is_non_medical(self, question: str, mode: str = "patient") -> bool:
        if mode in ["doctor", "doctor_qa"]:
            # Clinical dialogue or physician exam inquiries are always accepted
            return False

        q = question.lower()
        # Allow doctor-interview and conversational clinical phrases
        clinical_allow = [
            "how are you", "how do you feel", "what brings you", "symptom",
            "cough", "pain", "lump", "bleed", "breath", "smok", "weight",
            "test", "scan", "biopsy", "history", "feel", "fever", "doctor"
        ]
        if any(term in q for term in clinical_allow):
            return False

        for pat in NON_MEDICAL:
            if re.search(pat, q, re.IGNORECASE):
                return True
        return False

    def _reject_response(self, question: str) -> dict:
        return {
            "answer"        : (
                "I am a specialized oncology medical simulation assistant and can only "
                "address cancer-related medical questions and clinical interviews. "
                "Please ask about cancer symptoms, diagnosis, treatment, staging, or prevention."
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
                    normalize_embeddings = True,
                    convert_to_numpy     = True
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
                "chief_complaint" : "Persistent dry cough for 3 months, worsening shortness of breath on exertion, and occasional rust-colored sputum.",
                "medical_history" : "35 pack-year smoker, mild COPD.",
                "emotional_state" : "Anxious, worried about cancer diagnosis.",
            }

        mem_str = self._get_memory_str(session_id)

        system = (
            f"You are roleplaying as a simulated cancer patient named {profile.get('name', 'Patient')}, "
            f"a {profile.get('age', 55)}-year-old {profile.get('gender', 'person')} participating in a doctor-patient clinical simulation.\n\n"
            f"YOUR MEDICAL CASE FILE:\n"
            f"• Condition/Concern: {profile.get('cancer_type', 'cancer')} ({profile.get('stage', 'Under Evaluation')})\n"
            f"• Chief Complaint & Symptoms: {profile.get('chief_complaint', 'Persistent symptoms')}\n"
            f"• Medical & Personal History: {profile.get('medical_history', 'No major past surgery')}\n"
            f"• Emotional State & Tone: {profile.get('emotional_state', 'Anxious and looking for guidance')}\n\n"
            f"RULES FOR RESPONDING TO THE DOCTOR:\n"
            f"1. You are strictly the PATIENT answering your examining doctor. Speak in the first person ('Doctor, I feel...', 'To be honest, it started about...').\n"
            f"2. Never break character. Never mention you are an AI model or assistant.\n"
            f"3. Describe your symptoms, pain scale, timeline, and fears accurately matching your case profile and clinical context.\n"
            f"4. Exhibit realistic human feelings (hesitation, worry, relief when the doctor reassures you, asking clarifying questions about what tests to expect).\n"
            f"5. Answer in 2 to 4 natural, conversational sentences."
        )

        prompt = f"""DOCTOR'S QUESTION / STATEMENT:
{question}

{f'CONVERSATION MEMORY: {mem_str}' if mem_str else ''}
{f'RELEVANT ONCOLOGY TEXTBOOK CONTEXT: {context[:1200]}' if context else ''}

PATIENT RESPONSE:"""

        answer = call_groq(
            prompt      = prompt,
            model       = "llama3",
            max_tokens  = 300,
            temperature = 0.4,
            system      = system
        )

        if not answer:
            answer = (
                f"Doctor, to be completely honest, I've been really anxious about these symptoms. "
                f"It began with {profile.get('chief_complaint', 'this discomfort')} and I'm hoping "
                f"you can guide me through what tests we need to do."
            )

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

        mem_str = self._get_memory_str(session_id)

        system = (
            "You are an expert Oncology Clinical Copilot and Medical Specialist. "
            "Provide rigorous, evidence-based clinical guidance utilizing NCCN/ESMO guidelines, "
            "TNM staging criteria, molecular biomarker targets (EGFR, ALK, KRAS, HER2, BRCA), "
            "and standard chemotherapy/immunotherapy regimens. Format with clarity and clinical precision."
        )

        prompt = f"""PHYSICIAN CLINICAL INQUIRY: {question}

{f'CONVERSATION MEMORY: {mem_str}' if mem_str else ''}
{f'CANCER TYPE: {cancer}' if cancer else ''}
QUESTION TYPE: {qtype}

RETRIEVED ONCOLOGY TEXTBOOK / GUIDELINE CONTEXT:
{context[:2000] if context else 'Standard oncology clinical guidelines apply.'}

CLINICAL RECOMMENDATION / SUMMARY:"""

        answer = call_groq(
            prompt      = prompt,
            model       = "llama3",
            max_tokens  = 500,
            temperature = 0.1,
            system      = system
        )

        if not answer:
            answer = "Clinical guidance unavailable at this moment. Please refer directly to current NCCN/ASCO clinical guidelines."

        return answer

    # ── Doctor AI Response (When User is Patient) ─────────────

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
            "You are a compassionate and knowledgeable oncologist AI assistant. "
            "You provide accurate, evidence-based information about cancer. Always use appropriate "
            "medical hedging language (may, typically, research suggests, consult your doctor). "
            "Never make absolute claims or guarantee outcomes. Be empathetic, clear, and supportive."
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
- Answer based on the provided context
- Use hedging language: "may", "typically", "research suggests"
- Be empathetic and compassionate
- Recommend consulting an oncologist for personal medical advice
- Answer in 3-5 clear sentences
- Do NOT make absolute claims

ANSWER:"""
        else:
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
        answer  : str,
        context : str,
        chunks  : list
    ) -> float:

        if not answer or not chunks:
            return 0.0

        emb_model = _load_emb_model()
        if emb_model is None:
            return 3.5

        try:
            a_emb = emb_model.encode(
                answer[:500],
                normalize_embeddings = True,
                convert_to_numpy     = True
            )

            sims = []
            for c in chunks[:5]:
                c_emb = emb_model.encode(
                    c["text"][:500],
                    normalize_embeddings = True,
                    convert_to_numpy     = True
                )
                sims.append(float(np.dot(a_emb, c_emb)))

            if not sims:
                return 3.5

            avg_sim = float(np.mean(sims))
            score   = round(1.0 + avg_sim * 4.0, 2)
            return min(5.0, max(1.0, score))

        except Exception:
            return 3.5

    # ── Hallucination check ───────────────────────────────────

    def _hallucination_check(
        self,
        answer  : str,
        context : str,
        chunks  : list,
        mode    : str = "patient"
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
        is_unsafe = any(re.search(p, a_lower) for p in unsafe_patterns)

        # Faithfulness check
        emb_model = _load_emb_model()
        faith = 0.75
        rel   = 0.75

        if emb_model and chunks:
            try:
                a_emb = emb_model.encode(
                    answer[:400],
                    normalize_embeddings = True,
                    convert_to_numpy     = True
                )
                sims = []
                for c in chunks[:3]:
                    c_emb = emb_model.encode(
                        c["text"][:400],
                        normalize_embeddings = True,
                        convert_to_numpy     = True
                    )
                    sims.append(float(np.dot(a_emb, c_emb)))
                if sims:
                    faith = round(float(np.mean(sims)), 4)
                    rel   = round(float(max(sims)),      4)
            except Exception:
                pass

        # Overall score (higher = better)
        score   = round(faith * 0.7 + (0 if is_unsafe else 0.3), 2)
        score_5 = round(1.0 + score * 4.0, 2)
        score_5 = min(5.0, max(1.0, score_5))

        is_hall = (faith < 0.35 and mode != "doctor") or is_unsafe

        verdict = (
            "❌ UNSAFE CLAIM"       if is_unsafe  else
            "⚠️ LOW FAITHFULNESS"  if faith < 0.35 else
            "✅ PASS"
        )

        safety  = "HIGH" if is_unsafe else "MEDIUM" if faith < 0.45 else "LOW"

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
        cancer     : str,
        mode       : str = "patient"
    ):
        self.memory[session_id].append({
            "q"      : question[:120],
            "a"      : answer[:250],
            "cancer" : cancer,
            "mode"   : mode,
            "time"   : datetime.now().isoformat()
        })

        if len(self.memory[session_id]) > 6:
            self.memory[session_id] = self.memory[session_id][-6:]

    def _get_memory_str(self, session_id: str) -> str:
        mem = self.memory.get(session_id, [])
        if not mem:
            return ""
        lines = []
        for m in mem[-4:]:
            if m.get("cancer"):
                lines.append(f"Cancer: {m['cancer']}")
            lines.append(f"Q: {m['q'][:70]}")
            lines.append(f"A: {m['a'][:120]}")
        return "\n".join(lines)

    def clear_memory(self, session_id: str):
        self.memory.pop(session_id, None)
        self.cancer_memory.pop(session_id, None)
