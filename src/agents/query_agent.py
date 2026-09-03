# src/agents/query_agent.py
# ============================================================
# NITPY Query Agent — Comprehensive Oncology AI
# Features:
#   1. Robust Memory Layer (Tracks cancer type, resolves "this", "it", "that")
#   2. Strict Doctor Response Rule: 1-Line Answer OR Clinical Question (Not Both)
#   3. Bi-directional Simulation (Patient Mode & Doctor Mode)
#   4. Dual LLM Backend: Groq API (Cloud) + Ollama (Local) + High-Fidelity Medical Fallbacks
#   5. Zero-RAM Streaming ChromaDB auto-extraction
# ============================================================

import os
import sys
import re
import json
import random
import requests
import numpy as np
from datetime import datetime
from collections import defaultdict

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
    from agents.groq_client import call_groq, get_groq_key
except ImportError:
    try:
        from src.agents.groq_client import call_groq, get_groq_key
    except ImportError:
        def get_groq_key(): return os.environ.get("GROQ_API_KEY", "")
        def call_groq(prompt, **kwargs): return ""

# ── Lazy globals ──────────────────────────────────────────────
_chroma_client = None
_collection    = None
_emb_model     = None


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
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    _ensure_chroma_extracted()
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(
            path=os.path.join(ROOT, "chroma_db")
        )
        _collection = _chroma_client.get_or_create_collection(
            name="medical_rag",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB loaded: {_collection.count()} chunks")
    except Exception as e:
        print(f"ChromaDB error: {e}")
        _collection = None
    return _collection


def _load_emb():
    global _emb_model
    if _emb_model is not None:
        return _emb_model
    try:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embedding model ready")
    except Exception as e:
        print(f"Embedding error: {e}")
        _emb_model = None
    return _emb_model


# ── Unified LLM Caller (Groq Cloud API -> Local Ollama -> Fallback) ──
def _call_llm(prompt: str, system: str = "You are an expert oncologist.", max_tokens: int = 150) -> str:
    """Unified LLM call: Prioritizes Groq API, falls back to Ollama."""
    # 1. Try Groq API if key exists
    if get_groq_key():
        resp = call_groq(prompt=prompt, system=system, max_tokens=max_tokens, temperature=0.1)
        if resp and not resp.startswith("⚠️"):
            return resp

    # 2. Try Local Ollama
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model"  : "llama3",
                "prompt" : f"System: {system}\n\nUser: {prompt}",
                "stream" : False,
                "options": {
                    "temperature" : 0.1,
                    "num_predict" : max_tokens,
                }
            },
            timeout=10
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            for tok in ["<|im_end|>","<|im_start|>","<|end|>","<|assistant|>","[/INST]","</s>"]:
                raw = raw.replace(tok, "")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw:
                return raw
    except Exception:
        pass

    return ""


# ── Canonical Cancers Dictionary ──────────────────────────────
CANCER_KEYWORDS = {
    "lung cancer"       : ["lung", "lungs", "nsclc", "sclc", "bronchial", "pulmonary"],
    "breast cancer"     : ["breast", "breasts", "mammary", "mastectomy", "lumpectomy", "her2", "er+", "pr+"],
    "colon cancer"      : ["colon", "colorectal", "bowel", "rectal", "rectum", "polyps", "colonoscopy"],
    "prostate cancer"   : ["prostate", "psa", "prostatic", "gleason"],
    "skin cancer"       : ["skin", "melanoma", "basal cell", "squamous cell", "mole"],
    "leukemia"          : ["leukemia", "leukaemia", "all", "aml", "cll", "cml", "blood cancer"],
    "lymphoma"          : ["lymphoma", "hodgkin", "non-hodgkin", "nhl", "lymph node"],
    "pancreatic cancer" : ["pancreas", "pancreatic", "whipple"],
    "ovarian cancer"    : ["ovary", "ovarian", "ca-125"],
    "cervical cancer"   : ["cervix", "cervical", "pap smear", "hpv"],
    "liver cancer"      : ["liver", "hepatic", "hcc", "hepatocellular"],
    "stomach cancer"    : ["stomach", "gastric"],
    "kidney cancer"     : ["kidney", "renal", "rcc"],
    "bladder cancer"    : ["bladder", "urothelial"],
    "thyroid cancer"    : ["thyroid", "papillary thyroid", "follicular thyroid"],
    "brain tumor"       : ["brain", "glioblastoma", "glioma", "astrocytoma"],
    "bone cancer"       : ["bone", "osteosarcoma", "ewing"],
    "esophageal cancer" : ["esophagus", "esophageal", "esophagus cancer"],
    "testicular cancer" : ["testis", "testicular", "seminoma"],
    "uterine cancer"    : ["uterus", "uterine", "endometrial"],
    "multiple myeloma"  : ["myeloma", "multiple myeloma", "plasma cell"],
}

CANCER_PRONOUNS = [
    "this cancer", "the cancer", "this disease", "the disease",
    "this condition", "this type", "this", "it", "that", "these",
    "the tumor", "this tumor"
]

NON_MEDICAL_PATTERNS = [
    r"\b(weather|temperature|rain|forecast)\b",
    r"\b(stock|bitcoin|crypto|shares|market)\b",
    r"\b(movie|film|actor|actress|song|music)\b",
    r"\b(recipe|restaurant|cook|dinner)\b",
    r"\b(cricket|football|sports|ipl|nba)\b",
    r"\b(python|javascript|coding|software|algorithm)\b",
    r"\b(politics|election|president|minister)\b",
]

QUESTION_INTENTS = {
    "symptoms"    : ["symptom", "sign", "feel", "warning", "early", "notice", "cough", "pain", "lump", "bleed", "fatigue"],
    "treatment"   : ["treat", "treatment", "therapy", "cure", "chemo", "chemotherapy", "radiation", "surgery", "drug", "medicine", "immunotherapy"],
    "survival"    : ["survive", "survival", "rate", "outlook", "prognosis", "live", "life", "chance", "years", "remission"],
    "diagnosis"   : ["diagnos", "detect", "test", "screening", "scan", "biopsy", "mri", "ct", "ultrasound", "blood test", "mammogram"],
    "stages"      : ["stage", "stages", "staging", "spread", "metastasis", "grade", "tnm", "advanced"],
    "causes"      : ["cause", "why", "risk", "reason", "hereditary", "genetic", "prevent", "lifestyle", "smoke", "smoking"],
    "side_effects": ["side effect", "reaction", "toxic", "nausea", "hair loss", "vomiting", "weakness"],
    "definition"  : ["what is", "define", "explain", "meaning", "overview", "tell me about"],
    "general"     : ["cancer", "help", "information", "consultation"]
}

# Clinical questions the doctor asks the patient
CLINICAL_QUESTIONS = {
    "symptoms": [
        "How long have you been experiencing these symptoms, and have they become more frequent?",
        "Have you noticed any unintentional weight loss, persistent fatigue, or night sweats?",
        "Are you currently having any localized pain, bleeding, or shortness of breath?",
        "When did you first notice these changes, and is anyone in your family affected by similar symptoms?",
    ],
    "treatment": [
        "Have you discussed your biopsy pathology or molecular test results with your oncologist yet?",
        "Are you currently undergoing any chemotherapy, radiation, or previous surgeries?",
        "Have your doctors outlined whether surgery or systemic chemotherapy will be the first step?",
        "What specific treatment options or clinical trials has your oncology care team proposed?",
    ],
    "survival": [
        "Do you know the exact stage and biomarker status (such as hormone receptors or PD-L1) of the tumor?",
        "Has a full body PET-CT scan or MRI been performed to confirm whether the disease is localized?",
        "Are you looking for 5-year survival statistics for localized, regional, or metastatic disease?",
        "How is your overall performance status and daily energy level during day-to-day activities?",
    ],
    "diagnosis": [
        "Have you already had a tissue biopsy or surgical excision performed to confirm the pathology?",
        "Have imaging scans such as a contrast CT, MRI, or PET scan been completed?",
        "Are you experiencing any new focal symptoms that prompted you to look into diagnostic tests?",
        "Did your physician recommend any specific blood biomarker screenings, such as CEA, CA-125, or PSA?",
    ],
    "stages": [
        "Do you know if nearby lymph nodes or other organs were involved on your latest scan?",
        "Has your medical team discussed whether the cancer is considered Stage I, II, III, or IV?",
        "Would you like to know how tumor size and lymph node spread determine the stage?",
        "Are you currently experiencing any symptoms suggestive of advanced or metastatic disease?",
    ],
    "causes": [
        "Do you have a personal history of tobacco use, environmental exposures, or family cancer history?",
        "Has anyone in your immediate family undergone genetic counseling or BRCA/Lynch testing?",
        "Are you interested in learning about modifiable lifestyle factors to lower overall recurrence risk?",
        "Did your physician suggest any genetic or hereditary risk evaluation for your family?",
    ],
    "side_effects": [
        "Which specific medications or radiation areas are causing you discomfort right now?",
        "Are you experiencing manageable nausea, severe fatigue, neuropathy, or appetite changes?",
        "Have you been prescribed antiemetics or supportive medications to help control your symptoms?",
        "Are you able to stay well-hydrated and maintain adequate caloric intake each day?",
    ],
    "general": [
        "What specific questions or concerns about cancer can I clarify for you today?",
        "Are you inquiring for yourself or supporting a loved one through a recent diagnosis?",
        "Would you like to focus on symptoms, diagnostic workups, or available treatment regimens?",
        "How are you coping emotionally, and do you have a strong support system around you?",
    ],
}

# Accurate 1-line medical reference fallbacks
FALLBACK_ANSWERS = {
    "lung cancer": {
        "symptoms"    : "Common symptoms of lung cancer include a persistent cough, coughing up blood, shortness of breath, chest pain, and unintentional weight loss.",
        "treatment"   : "Treatment for lung cancer involves surgery, chemotherapy, radiation therapy, targeted therapies (EGFR/ALK inhibitors), and immune checkpoint inhibitors.",
        "survival"    : "The overall 5-year survival rate for lung cancer is approximately 25%, ranging from over 60% for localized disease to 8% for metastatic disease.",
        "diagnosis"   : "Lung cancer is diagnosed through low-dose chest CT imaging followed by bronchoscopy or CT-guided core needle biopsy for histopathological confirmation.",
        "stages"      : "Lung cancer is staged from Stage I (localized tumor) to Stage IV (metastatic spread to distant organs or both lungs).",
        "causes"      : "Cigarette smoking is the primary cause of lung cancer, accounting for 85% of cases, along with radon exposure and genetic mutations.",
        "side_effects": "Lung cancer treatments can cause fatigue, shortness of breath, nausea, pneumonitis, and decreased appetite.",
        "definition"  : "Lung cancer is a malignant neoplasm of the pulmonary tissues, categorized primarily as non-small cell lung cancer (85%) or small cell lung cancer (15%).",
    },
    "breast cancer": {
        "symptoms"    : "Breast cancer typically presents as a painless firm breast lump, skin dimpling, nipple retraction, or spontaneous nipple discharge.",
        "treatment"   : "Breast cancer treatment includes lumpectomy or mastectomy, radiation, chemotherapy, HER2-targeted therapy (trastuzumab), and endocrine therapy.",
        "survival"    : "The 5-year relative survival rate for localized breast cancer is 99%, with an overall 5-year survival rate of approximately 91%.",
        "diagnosis"   : "Breast cancer is diagnosed using diagnostic mammography, ultrasound, and image-guided core needle biopsy with receptor testing.",
        "stages"      : "Breast cancer stages range from Stage 0 (DCIS) through Stage IV (metastatic spread to bones, liver, lungs, or brain).",
        "causes"      : "Risk factors include age, BRCA1/BRCA2 genetic mutations, family history, hormone exposure, and dense breast tissue.",
        "side_effects": "Common treatment side effects include fatigue, lymphedema, hot flashes, hair loss, and mild cardiac changes from targeted drugs.",
        "definition"  : "Breast cancer is a malignancy originating in the lobules or ducts of the mammary gland tissue.",
    },
    "colon cancer": {
        "symptoms"    : "Colorectal cancer symptoms include rectal bleeding, dark stools, changes in bowel habits, iron-deficiency anemia, and abdominal cramping.",
        "treatment"   : "Treatment includes surgical resection (colectomy), adjuvant FOLFOX/CAPOX chemotherapy, and targeted anti-VEGF or anti-EGFR therapies.",
        "survival"    : "The 5-year relative survival rate is 91% for localized colon cancer and 65% across all stages combined.",
        "diagnosis"   : "Colon cancer is diagnosed through screening colonoscopy with tissue biopsy and baseline CEA tumor marker testing.",
        "stages"      : "Colon cancer is staged from Stage I (confined to bowel wall) to Stage IV (metastasis to liver or peritoneum).",
        "causes"      : "Causes include adenomatous polyps, inherited Lynch syndrome, inflammatory bowel disease, and high-fat, low-fiber diets.",
        "side_effects": "Treatments may cause temporary bowel irregularity, peripheral neuropathy from oxaliplatin, fatigue, and mild nausea.",
        "definition"  : "Colon cancer is a malignant adenocarcinoma arising from the epithelial lining of the large intestine.",
    },
    "prostate cancer": {
        "symptoms"    : "Prostate cancer often has no early symptoms, but can cause urinary hesitancy, weak stream, nocturia, and bone pain in advanced stages.",
        "treatment"   : "Treatment options include active surveillance, radical prostatectomy, radiation therapy, and androgen deprivation therapy (ADT).",
        "survival"    : "The 5-year relative survival rate for localized and regional prostate cancer is nearly 100%.",
        "diagnosis"   : "Diagnosis involves serum PSA measurement, digital rectal examination (DRE), and multiparametric MRI-guided prostate biopsy.",
        "stages"      : "Staging is determined by the TNM classification system and Gleason Grade Group (1 to 5).",
        "causes"      : "Primary risk factors include advancing age, African ancestry, and inherited mutations in BRCA2 or HOXB13 genes.",
        "side_effects": "Treatments can result in erectile dysfunction, urinary incontinence, and hot flashes from hormone therapy.",
        "definition"  : "Prostate cancer is a malignant adenocarcinoma developing within the male prostate gland.",
    }
}


# ============================================================
# QUERY AGENT CLASS
# ============================================================

class QueryAgent:
    """
    Oncology QA Agent with Session Memory, Coreference Resolution,
    and Strict 1-Line Answer vs Clinical Question generation.
    """

    def __init__(self):
        self.memory = defaultdict(lambda: {
            "cancer_type"    : "",
            "cancer_history" : [],
            "qa_history"     : [],
            "last_intent"    : "general",
            "turn_count"     : 0,
            "answered"       : [],
        })
        print("QueryAgent initialized (Ready for Groq / Local RAG & Simulation)")

    # ==========================================================
    # MAIN PROCESS FUNCTION
    # ==========================================================

    def process(
        self,
        question              : str,
        session_id            : str  = "default",
        mode                  : str  = "patient",
        doctor_response_style : str  = "auto",   # 'auto', 'answer_only', 'question_only'
        patient_profile       : dict = None,
        **kwargs
    ) -> dict:
        """
        Main entry point for processing patient or doctor inquiries.
        Accepts all expected arguments with full backward compatibility.
        """

        mem = self.memory[session_id]

        # ── 1. Non-medical rejection ──────────────────────────
        if mode == "patient" and self._is_non_medical(question):
            return self._non_medical_response()

        # ── 2. Detect / update cancer type ────────────────────
        if mode == "doctor" and patient_profile:
            cancer = patient_profile.get("cancer_type", "lung cancer")
            mem["cancer_type"] = cancer
        else:
            cancer = self._detect_cancer(question, mem)

        # ── 3. Resolve pronouns using memory ("this" -> cancer) ──
        resolved_q = self._resolve_pronouns(question, mem, mode=mode)

        # ── 4. Detect clinical intent ─────────────────────────
        intent = self._detect_intent(resolved_q)

        # ── 5. Retrieve from ChromaDB ─────────────────────────
        chunks, context = self._retrieve(resolved_q, cancer)

        # ── 6. Handle Different Simulation Modes ──────────────
        if mode == "doctor":
            # AI is Simulated Patient
            response_text = self._generate_patient_response(
                question   = question,
                context    = context,
                profile    = patient_profile or {},
                mem        = mem
            )
            response_type = "patient_reply"

        elif mode in ["doctor_qa", "copilot"]:
            # AI is Clinical Specialist Copilot
            response_text = self._generate_clinical_response(
                question = resolved_q,
                context  = context,
                cancer   = cancer,
                intent   = intent,
                mem      = mem
            )
            response_type = "clinical_guidance"

        else:
            # Mode = Patient (AI is Doctor)
            # Rule: Either 1-Line Answer OR Clinical Question to Patient (Not Both)
            if doctor_response_style == "answer_only":
                should_ask = False
            elif doctor_response_style == "question_only":
                should_ask = True
            else:
                should_ask = self._should_ask_question(mem, intent, question)

            if should_ask:
                response_text = self._get_clinical_question(intent, cancer, mem)
                response_type = "question"
            else:
                response_text = self._generate_answer(
                    resolved_q, context, chunks, cancer, intent, mem
                )
                response_type = "answer"

        # ── 7. Score and Hallucination Check ──────────────────
        confidence = self._score(response_text, chunks)
        hall       = self._hall_check(response_text, chunks, response_type)

        # ── 8. Update Memory Layer ────────────────────────────
        self._update_memory(
            mem           = mem,
            question      = question,
            resolved_q    = resolved_q,
            response      = response_text,
            cancer        = cancer,
            intent        = intent,
            response_type = response_type
        )

        # ── 9. Build sources ──────────────────────────────────
        sources = [
            {
                "source" : c.get("source", "unknown"),
                "score"  : round(c.get("rerank_score", 0), 3)
            }
            for c in chunks[:3]
        ]

        was_resolved = resolved_q.strip().lower() != question.strip().lower()

        return {
            "answer"            : response_text,
            "response_type"     : response_type,       # "answer" or "question"
            "action"            : response_type,       # ui.py compatibility
            "question"          : question,
            "resolved_question" : resolved_q,
            "was_resolved"      : was_resolved,
            "question_type"     : intent,
            "cancer_type"       : cancer,
            "is_followup"       : self._is_followup(question),
            "confidence"        : confidence,
            "sources"           : sources,
            "hallucination"     : hall,
            "memory_context"    : self._get_memory_summary(mem),
            "chunks_used"       : len(chunks),
            "turn_count"        : mem["turn_count"],
            "mode"              : mode,
        }

    # ==========================================================
    # CANCER DETECTION & MEMORY RESOLUTION
    # ==========================================================

    def _detect_cancer(self, question: str, mem: dict) -> str:
        """Detect cancer type from current question or fallback to memory."""
        q = question.lower()
        for cancer, keywords in CANCER_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return cancer
        return mem.get("cancer_type", "")

    def _resolve_pronouns(self, question: str, mem: dict, mode: str = "patient") -> str:
        """
        Resolves 'this', 'it', 'that', 'this cancer' to stored cancer type in patient mode.
        Example: 'what is the treatment for this?' -> 'what is the treatment for lung cancer?'
        """
        if mode == "doctor":
            return question

        cancer = mem.get("cancer_type", "")
        if not cancer:
            return question

        q = question.strip()
        explicit_pats = [
            (r"this cancer",    cancer),
            (r"the cancer",     cancer),
            (r"this disease",   cancer),
            (r"this condition", cancer),
            (r"this type",      cancer),
            (r"this tumor",     cancer),
        ]

        for pat, rep in explicit_pats:
            if re.search(pat, q, re.IGNORECASE):
                resolved = re.sub(pat, rep, q, count=1, flags=re.IGNORECASE)
                if resolved.lower() != q.lower():
                    print(f"  🧠 Memory Resolved: '{question}' → '{resolved}'")
                    return resolved

        contextual_pats = [
            (r"(for|of|with|about|causes?|treating|diagnose)\s+this", r" " + cancer),
            (r"(for|of|with|about|causes?|treating|diagnose)\s+it",   r" " + cancer),
            (r"(for|of|with|about|causes?|treating|diagnose)\s+that", r" " + cancer),
            (r"(is|can)\s+this", r" " + cancer),
            (r"(is|can)\s+it",   r" " + cancer),
            (r"(is|can)\s+that", r" " + cancer),
            (r"what about this", "what about " + cancer),
            (r"what about it",   "what about " + cancer),
        ]

        for pat, rep in contextual_pats:
            if re.search(pat, q, re.IGNORECASE):
                resolved = re.sub(pat, rep, q, count=1, flags=re.IGNORECASE)
                if resolved.lower() != q.lower():
                    print(f"  🧠 Memory Resolved: '{question}' → '{resolved}'")
                    return resolved

        return question

    def _detect_intent(self, question: str) -> str:
        """Detect what the patient is asking about."""
        q = question.lower()
        scores = {}
        for intent, keywords in QUESTION_INTENTS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general"
        return max(scores, key=scores.get)

    def _should_ask_question(self, mem: dict, intent: str, question: str) -> bool:
        """
        Decide whether the Doctor AI should give an answer or ask a clinical question.
        - If patient shares personal symptoms/discomfort -> Ask targeted clinical question.
        - If patient asks direct medical/treatment question -> Give 1-line answer.
        - Every 3rd turn in an ongoing dialogue -> Ask clinical question for engagement.
        """
        q_low = question.lower()
        symptom_cues = ["i feel", "i have", "i'm having", "i noticed", "i cough", "i bleed", "scared", "worried", "pain in my"]
        if any(cue in q_low for cue in symptom_cues):
            return True

        turn = mem.get("turn_count", 0)
        if turn == 0:
            return False

        if intent in mem.get("answered", []) and intent != "general":
            return True

        if turn % 3 == 2:
            return True

        return False

    # ==========================================================
    # CHROMADB RETRIEVAL
    # ==========================================================

    def _retrieve(self, question: str, cancer: str = "", k: int = 5) -> tuple:
        """Retrieve relevant chunks from ChromaDB."""
        col = _load_chroma()
        emb = _load_emb()

        if col is None or emb is None:
            return [], ""

        query = question
        if cancer and cancer not in question.lower():
            query = f"{cancer} {question}"

        try:
            q_emb = emb.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            result = col.query(
                query_embeddings=[q_emb.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            chunks = []
            for i in range(len(result["ids"][0])):
                text   = result["documents"][0][i]
                src    = result["metadatas"][0][i].get("source", "?")
                raw    = 1 - result["distances"][0][i]
                c_emb  = emb.encode(
                    text[:500],
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                rerank = float(np.dot(q_emb, c_emb))

                chunks.append({
                    "text"        : text,
                    "source"      : src,
                    "raw_score"   : round(raw, 4),
                    "rerank_score": round(rerank, 4),
                })

            chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            context = "\n\n".join([c["text"] for c in chunks[:k]])
            return chunks, context

        except Exception as e:
            print(f"Retrieval error: {e}")
            return [], ""

    # ==========================================================
    # GENERATE DIRECT ANSWER (Strict 1-Line)
    # ==========================================================

    def _generate_answer(
        self,
        question : str,
        context  : str,
        chunks   : list,
        cancer   : str,
        intent   : str,
        mem      : dict
    ) -> str:
        """
        Generate EXACTLY ONE accurate, concise answer sentence.
        Strictly no trailing questions.
        """
        mem_ctx = self._get_memory_summary(mem)

        system = (
            "You are an expert oncologist AI. Provide an accurate, direct ONE-LINE answer "
            "to the patient's cancer question, formatted like a clean QA JSON dataset entry. "
            "STRICT INSTRUCTION: Provide exactly ONE clear, concise medical sentence. "
            "DO NOT ask any question. Do not include follow-up questions."
        )

        if context:
            prompt = f"""PATIENT QUESTION: {question}
{f'CANCER TOPIC: {cancer}' if cancer else ''}
{f'HISTORY: {mem_ctx}' if mem_ctx else ''}

MEDICAL CONTEXT:
{context[:1500]}

RULES:
1. Give EXACTLY ONE sentence answer.
2. Base answer strictly on medical facts.
3. Do NOT ask any question.
4. Do NOT say 'based on the context'.

ONE-LINE ANSWER:"""
        else:
            prompt = f"""PATIENT QUESTION: {question}
{f'CANCER TOPIC: {cancer}' if cancer else ''}

Give EXACTLY ONE medically accurate sentence answering this question. Do NOT ask any question back.
ONE-LINE ANSWER:"""

        answer = _call_llm(prompt=prompt, system=system, max_tokens=100)

        # Clean answer to enforce single statement
        if answer:
            sentences = re.split(r'(?<=[.!])\s+', answer.strip())
            clean_s = [s.strip() for s in sentences if not s.strip().endswith("?")]
            if clean_s:
                answer = clean_s[0].strip()
            else:
                answer = sentences[0].rstrip("?") + "."

        if not answer or len(answer.strip()) < 10:
            answer = self._get_fallback_answer(intent, cancer)

        return answer

    # ==========================================================
    # GET CLINICAL QUESTION (Strict Question Only)
    # ==========================================================

    def _get_clinical_question(self, intent: str, cancer: str, mem: dict) -> str:
        """Get a clinically appropriate question to ask the patient."""
        c = cancer.capitalize() if cancer else "your condition"
        questions = CLINICAL_QUESTIONS.get(intent, CLINICAL_QUESTIONS["general"])

        # Try generating via LLM first
        mem_ctx = self._get_memory_summary(mem)
        system = (
            "You are an empathetic oncologist consulting with a patient. "
            "Ask the patient exactly ONE relevant clinical follow-up question. "
            "STRICT INSTRUCTION: Output ONLY the question ending with a question mark. "
            "DO NOT give an answer or medical explanation."
        )
        prompt = f"""PATIENT CONTEXT: {mem_ctx}
TOPIC: {c} ({intent})

Ask the patient exactly ONE targeted clinical question about their symptoms, history, or tests.
DOCTOR'S QUESTION:"""

        q_llm = _call_llm(prompt=prompt, system=system, max_tokens=60)
        if q_llm and "?" in q_llm:
            questions_found = re.findall(r'([^.!?]*\?)', q_llm)
            if questions_found:
                return questions_found[0].strip()

        # Fallback to curated clinical question list
        turn = mem.get("turn_count", 0)
        idx  = turn % len(questions)
        return questions[idx]

    # ==========================================================
    # PATIENT AI ROLEPLAY (Doctor Mode)
    # ==========================================================

    def _generate_patient_response(
        self,
        question : str,
        context  : str,
        profile  : dict,
        mem      : dict
    ) -> str:
        """Simulate a patient answering the examining doctor."""
        if not profile:
            profile = {
                "name"            : "Robert Miller",
                "age"             : 58,
                "gender"          : "Male",
                "cancer_type"     : "lung cancer",
                "stage"           : "Suspected NSCLC",
                "chief_complaint" : "Persistent cough for 3 months, fatigue, rust-colored sputum.",
                "medical_history" : "35 pack-year smoker.",
                "emotional_state" : "Worried about cancer.",
            }

        mem_ctx = self._get_memory_summary(mem)

        system = (
            f"You are roleplaying as a simulated cancer patient named {profile.get('name', 'Patient')}, "
            f"a {profile.get('age', 55)}-year-old {profile.get('gender', 'person')} consulting your examining doctor.\n"
            f"PROFILE:\n"
            f"• Condition: {profile.get('cancer_type', 'cancer')} ({profile.get('stage', 'Under Evaluation')})\n"
            f"• Symptoms: {profile.get('chief_complaint', 'Symptoms')}\n"
            f"• History: {profile.get('medical_history', 'None')}\n"
            f"• Tone: {profile.get('emotional_state', 'Anxious')}\n\n"
            f"RULES:\n"
            f"1. Speak in first person as the patient ('Doctor, I feel...').\n"
            f"2. Answer the doctor's specific inquiry in 2-3 natural sentences."
        )

        prompt = f"""DOCTOR'S INQUIRY: {question}
{f'PRIOR DIALOGUE: {mem_ctx}' if mem_ctx else ''}

PATIENT'S RESPONSE:"""

        ans = _call_llm(prompt=prompt, system=system, max_tokens=180)
        if not ans:
            ans = f"Doctor, I've had {profile.get('chief_complaint', 'these symptoms')} and I'm really hoping you can help me understand what is going on."
        return ans

    # ==========================================================
    # CLINICAL COPILOT (Doctor QA Mode)
    # ==========================================================

    def _generate_clinical_response(
        self,
        question : str,
        context  : str,
        cancer   : str,
        intent   : str,
        mem      : dict
    ) -> str:
        """Provide specialist-level oncological reference for physicians."""
        system = (
            "You are an Oncology Clinical Decision Support Copilot. Provide rigorous, evidence-based "
            "guidance referencing NCCN/ESMO guidelines, TNM staging, molecular biomarkers, and regimens."
        )
        prompt = f"""PHYSICIAN INQUIRY: {question}
{f'CANCER TOPIC: {cancer}' if cancer else ''}

RETRIEVED ONCOLOGY CONTEXT:
{context[:1800] if context else 'Standard oncology clinical guidelines apply.'}

CLINICAL GUIDANCE:"""

        ans = _call_llm(prompt=prompt, system=system, max_tokens=300)
        if not ans:
            ans = self._get_fallback_answer(intent, cancer)
        return ans

    # ==========================================================
    # FALLBACK ANSWERS
    # ==========================================================

    def _get_fallback_answer(self, intent: str, cancer: str) -> str:
        c = cancer.lower() if cancer else "cancer"
        if c in FALLBACK_ANSWERS:
            return FALLBACK_ANSWERS[c].get(
                intent,
                FALLBACK_ANSWERS[c].get("definition", f"Treatment and diagnosis of {c} depends on tumor staging and individual patient health.")
            )

        generic = {
            "symptoms"    : f"Common symptoms of {c} include unexplained weight loss, fatigue, persistent pain, and localized swelling or lumps.",
            "treatment"   : f"Treatment for {c} typically involves a multidisciplinary approach including surgery, chemotherapy, radiation, or targeted therapy.",
            "survival"    : f"Survival rates for {c} depend significantly on the stage at diagnosis, tumor biology, and overall health.",
            "diagnosis"   : f"Diagnosis of {c} requires imaging studies (CT/MRI/PET) followed by tissue biopsy for definitive histopathology.",
            "stages"      : f"{c.capitalize()} is staged from Stage I to Stage IV based on tumor size, lymph node involvement, and metastasis.",
            "causes"      : f"Risk factors for {c} include genetic mutations, environmental carcinogens, age, and lifestyle factors.",
            "side_effects": f"Treatments for {c} can cause temporary fatigue, nausea, appetite changes, and low blood cell counts.",
            "definition"  : f"{c.capitalize()} is a malignant disease characterized by uncontrolled cellular proliferation and potential to invade tissues.",
            "general"     : f"Management of {c} is tailored based on clinical staging, pathology, and molecular biomarker testing.",
        }
        return generic.get(intent, generic["general"])

    # ==========================================================
    # NON-MEDICAL CHECK
    # ==========================================================

    def _is_non_medical(self, question: str) -> bool:
        q = question.lower()
        clinical_allow = [
            "how are you", "how do you feel", "what brings you", "symptom",
            "cough", "pain", "lump", "bleed", "breath", "smok", "weight",
            "test", "scan", "biopsy", "history", "feel", "fever", "doctor",
            "this", "it", "treatment", "stage", "cure", "survive", "chemo"
        ]
        if any(term in q for term in clinical_allow):
            return False

        return any(re.search(p, q, re.IGNORECASE) for p in NON_MEDICAL_PATTERNS)

    def _non_medical_response(self) -> dict:
        return {
            "answer"            : "I specialize in oncology only. Please ask me about cancer symptoms, diagnosis, treatment, staging, or prognosis.",
            "response_type"     : "reject",
            "action"            : "reject",
            "question_type"     : "non_medical",
            "cancer_type"       : "",
            "is_followup"       : False,
            "was_resolved"      : False,
            "resolved_question" : "",
            "confidence"        : 0.0,
            "sources"           : [],
            "hallucination"     : {
                "score": 0.0, "verdict": "NON_MEDICAL",
                "safety": "LOW", "is_hallucinated": False,
                "faithfulness": 0.0, "relevance": 0.0,
            },
            "memory_context"    : "",
            "chunks_used"       : 0,
            "turn_count"        : 0,
        }

    # ==========================================================
    # FOLLOW-UP DETECTION
    # ==========================================================

    def _is_followup(self, question: str) -> bool:
        q = question.lower().strip()
        return any([
            any(p in q for p in CANCER_PRONOUNS),
            q.startswith("what about"),
            q.startswith("and "),
            q.startswith("also "),
            q.startswith("how about"),
            len(q.split()) <= 6,
        ])

    # ==========================================================
    # SCORING AND HALLUCINATION CHECK
    # ==========================================================

    def _score(self, answer: str, chunks: list) -> float:
        if not answer or not chunks:
            return 3.5
        emb = _load_emb()
        if emb is None:
            return 3.5
        try:
            a_emb = emb.encode(answer[:400], normalize_embeddings=True, convert_to_numpy=True)
            sims = [float(np.dot(a_emb, emb.encode(c["text"][:400], normalize_embeddings=True, convert_to_numpy=True))) for c in chunks[:5]]
            avg = float(np.mean(sims)) if sims else 0.5
            return round(min(5.0, max(1.0, 1.0 + avg * 4.0)), 2)
        except Exception:
            return 3.5

    def _hall_check(self, answer: str, chunks: list, response_type: str) -> dict:
        if response_type == "question":
            return {
                "score": 5.0, "verdict": "✅ CLINICAL QUESTION",
                "safety": "LOW", "is_hallucinated": False,
                "faithfulness": 1.0, "relevance": 1.0,
            }

        if not answer:
            return {
                "score": 0.0, "verdict": "EMPTY",
                "safety": "HIGH", "is_hallucinated": True,
                "faithfulness": 0.0, "relevance": 0.0,
            }

        unsafe = [
            r"100\s*%\s*(cure|cured|effective|guaranteed)",
            r"guaranteed\s+(cure|recovery|treatment)",
            r"definitely\s+(cures|eliminates|kills)",
            r"no\s+side\s+effects\s+at\s+all",
            r"miracle\s+(cure|drug|treatment)",
        ]
        is_unsafe = any(re.search(p, answer.lower()) for p in unsafe)
        faith = 0.75
        rel   = 0.75

        emb = _load_emb()
        if emb and chunks:
            try:
                a_emb = emb.encode(answer[:400], normalize_embeddings=True, convert_to_numpy=True)
                sims = [float(np.dot(a_emb, emb.encode(c["text"][:400], normalize_embeddings=True, convert_to_numpy=True))) for c in chunks[:3]]
                if sims:
                    faith = round(float(np.mean(sims)), 4)
                    rel   = round(float(max(sims)), 4)
            except Exception:
                pass

        score   = round(min(5.0, max(1.0, 1.0 + faith * 4.0)), 2)
        is_h    = faith < 0.35 or is_unsafe
        verdict = "❌ UNSAFE" if is_unsafe else ("⚠️ LOW FAITHFULNESS" if faith < 0.40 else "✅ PASS")
        safety  = "HIGH" if is_unsafe else ("MEDIUM" if faith < 0.5 else "LOW")

        return {
            "score"          : score,
            "verdict"        : verdict,
            "safety"         : safety,
            "is_hallucinated": bool(is_h),
            "faithfulness"   : faith,
            "relevance"      : rel,
        }

    # ==========================================================
    # MEMORY MANAGEMENT
    # ==========================================================

    def _update_memory(
        self, mem, question, resolved_q,
        response, cancer, intent, response_type
    ):
        mem["turn_count"] += 1
        if cancer:
            mem["cancer_type"] = cancer
            if cancer not in mem["cancer_history"]:
                mem["cancer_history"].append(cancer)

        mem["last_intent"] = intent
        if response_type == "answer" and intent not in mem["answered"]:
            mem["answered"].append(intent)

        mem["qa_history"].append({
            "turn"      : mem["turn_count"],
            "question"  : question,
            "resolved"  : resolved_q,
            "response"  : response[:150],
            "cancer"    : cancer,
            "intent"    : intent,
            "type"      : response_type,
            "timestamp" : datetime.now().isoformat()
        })

        if len(mem["qa_history"]) > 10:
            mem["qa_history"] = mem["qa_history"][-10:]

    def _get_memory_summary(self, mem: dict) -> str:
        lines = []
        if mem.get("cancer_type"):
            lines.append(f"Current topic: {mem['cancer_type']}")
        if mem.get("answered"):
            lines.append(f"Already covered: {', '.join(mem['answered'][-3:])}")

        history = mem.get("qa_history", [])
        for h in history[-2:]:
            if h.get("question"):
                lines.append(f"Patient: {h['question'][:60]}")
            if h.get("response") and h.get("type") == "answer":
                lines.append(f"Doctor: {h['response'][:80]}")

        return "\n".join(lines) if lines else ""

    def clear_memory(self, session_id: str):
        if session_id in self.memory:
            del self.memory[session_id]
