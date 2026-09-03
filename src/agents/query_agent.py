# src/agents/query_agent.py
# ============================================================
# NITPY Query Agent — Precision Oncology Consultation Engine
# Strict Architectural Rules:
#   1. SINGLE ACTION PER TURN: Either Answer OR Question (Never Both)
#   2. NO REPETITION on short replies ('no', 'yes', 'thanks', 'ok')
#   3. Contextual Memory & Coreference Resolution ("this" -> active cancer)
#   4. Patient Medical Notes & Tailored Precautions
#   5. Bi-directional Simulation (Doctor Mode & Patient Mode)
#   6. Zero-RAM Streaming ChromaDB auto-extraction (Render 512MB RAM safe)
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

# Automatic .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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


# ── Unified LLM Caller (Groq Cloud API -> Local Ollama -> None) ──
def _call_llm(prompt: str, system: str = "You are an expert oncologist.", max_tokens: int = 150) -> str:
    """Unified LLM call: Prioritizes Groq API, falls back to Ollama."""
    key = get_groq_key()
    if key:
        resp = call_groq(prompt=prompt, system=system, max_tokens=max_tokens, temperature=0.1)
        if resp and not resp.startswith("⚠️") and "API key" not in resp:
            return resp

    # Fallback to local Ollama
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
            timeout=8
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


# ── Canonical Cancer Regex Mappings (Whole-word boundaries) ──
CANCER_REGEX_MAP = [
    ("lung cancer",       r"\b(lung|lungs|nsclc|sclc|pulmonary neoplasm|pulmonary cancer)\b"),
    ("breast cancer",     r"\b(breast|breasts|mammary|mastectomy|lumpectomy|her2|er\+|pr\+)\b"),
    ("colon cancer",      r"\b(colon|colorectal|bowel|rectal|rectum|polyps|colonoscopy)\b"),
    ("prostate cancer",   r"\b(prostate|psa|prostatic|gleason)\b"),
    ("brain tumor",       r"\b(brain|brain tumor|brain cancer|glioblastoma|glioma|astrocytoma|meningioma)\b"),
    ("skin cancer",       r"\b(skin cancer|melanoma|basal cell|squamous cell carcinoma)\b"),
    ("pancreatic cancer", r"\b(pancreas|pancreatic|whipple)\b"),
    ("ovarian cancer",    r"\b(ovary|ovarian|ca-125)\b"),
    ("cervical cancer",   r"\b(cervix|cervical|pap smear|hpv)\b"),
    ("liver cancer",      r"\b(liver|hepatic|hcc|hepatocellular)\b"),
    ("stomach cancer",    r"\b(stomach|gastric)\b"),
    ("kidney cancer",     r"\b(kidney|renal|rcc)\b"),
    ("bladder cancer",    r"\b(bladder|urothelial)\b"),
    ("thyroid cancer",    r"\b(thyroid|papillary thyroid|follicular thyroid)\b"),
    ("bone cancer",       r"\b(bone cancer|osteosarcoma|ewing)\b"),
    ("esophageal cancer", r"\b(esophagus|esophageal)\b"),
    ("testicular cancer", r"\b(testis|testicular|seminoma)\b"),
    ("uterine cancer",    r"\b(uterus|uterine|endometrial)\b"),
    ("multiple myeloma",  r"\b(myeloma|multiple myeloma|plasma cell)\b"),
    ("lymphoma",          r"\b(lymphoma|hodgkin|non-hodgkin|nhl)\b"),
    ("leukemia",          r"\b(leukemia|leukaemia|aml|cll|cml|acute lymphoblastic leukemia|blood cancer)\b"),
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

# Conversational Pleasantries / Affirmations / Short Replies
ACK_PATTERNS = [
    r"^(ooh\s+|oh\s+)?(thanks|thank\s+you|thx|thanks\s+doctor|thank\s+you\s+so\s+much)[\.\!]?$",
    r"^(ok|okay|oohk|ohk|k|got\s+it|understood|alright|fine)[\.\!]?$",
    r"^(good|great|perfect|nice|cool)[\.\!]?$",
]

AFFIRMATION_PATTERNS = [
    r"^(yes|yeah|yep|yup|yes\s+all|all|all\s+of\s+them|both|yes\s+please|tell\s+me\s+all)[\.\!]?$",
]

NEGATION_PATTERNS = [
    r"^(no|nope|nah|not\s+really|none|not\s+yet|just\s+asking|just\s+information)[\.\!]?$",
]

QUESTION_INTENTS = {
    "symptoms"    : ["symptom", "sign", "feel", "warning", "early", "notice", "cough", "pain", "lump", "bleed", "fatigue", "fever", "headache", "seizure", "nausea"],
    "treatment"   : ["treat", "treatment", "therapy", "chemo", "chemotherapy", "radiation", "surgery", "drug", "medicine", "immunotherapy", "resection"],
    "recovery"    : ["recover", "recovery", "heal", "remission", "curable", "cured", "rehab", "get better", "overcome", "survive quickly"],
    "survival"    : ["survive", "survival", "rate", "outlook", "prognosis", "live", "life", "chance", "years", "mortality"],
    "diagnosis"   : ["diagnos", "detect", "test", "screening", "scan", "biopsy", "mri", "ct", "ultrasound", "blood test", "mammogram", "eeg"],
    "stages"      : ["stage", "stages", "staging", "spread", "metastasis", "grade", "tnm", "advanced", "benign", "malignant"],
    "causes"      : ["cause", "why", "risk", "reason", "hereditary", "genetic", "prevent", "lifestyle", "smoke", "smoking", "radiation exposure"],
    "side_effects": ["side effect", "reaction", "toxic", "hair loss", "vomiting", "weakness", "neuropathy"],
    "definition"  : ["what is", "define", "explain", "meaning", "overview", "tell me about", "what are"],
    "general"     : ["cancer", "help", "information", "consultation"]
}

# Curated Clinical Oncology Answers & Precautions (Strictly 1-Line Statements)
CLINICAL_KNOWLEDGE = {
    "lung cancer": {
        "definition"  : "Lung cancer is a malignant neoplasm of the pulmonary tissues, categorized primarily as non-small cell lung cancer (85%) or small cell lung cancer (15%).",
        "symptoms"    : "Common symptoms include a persistent or changing cough, coughing up blood (hemoptysis), pleuritic chest pain, shortness of breath, and unintentional weight loss.",
        "treatment"   : "Treatment involves surgical resection (lobectomy) for early stages, followed by platinum-based chemotherapy, precision stereotactic radiation, and targeted EGFR/ALK or immune checkpoint inhibitors.",
        "recovery"    : "Recovery depends on early surgical excision, pulmonary rehabilitation, smoking cessation, adequate caloric intake, and adjuvant targeted therapies to prevent recurrence.",
        "precautions" : "Avoid tobacco smoke and environmental dust, request an in-person chest CT scan, and seek emergency care if you experience hemoptysis or severe shortness of breath.",
        "question"    : "Are you or a loved one currently experiencing any persistent cough, chest discomfort, shortness of breath, or blood in phlegm?"
    },
    "brain tumor": {
        "definition"  : "A brain tumor is an abnormal growth of cells within the cranial cavity, arising from glial cells (gliomas, glioblastomas), meninges (meningiomas), or metastatic spread from other organs.",
        "symptoms"    : "Primary symptoms include persistent new-onset morning headaches, unexplained seizures, nausea/vomiting, and focal neurological deficits like limb weakness or vision changes.",
        "treatment"   : "Standard management involves maximum safe microsurgical resection, followed by targeted external beam radiation therapy and adjuvant temozolomide chemotherapy.",
        "recovery"    : "Recovery focuses on speech/physical neuro-rehabilitation, seizure prevention with antiepileptic medications, corticosteroid taper for edema, and serial contrast MRI monitoring.",
        "precautions" : "Avoid driving or swimming alone if seizures occur, take prescribed antiepileptic medications regularly, and seek emergency evaluation if acute weakness occurs.",
        "question"    : "Have you noticed any new-onset morning headaches, vision changes, seizures, or limb weakness recently?"
    },
    "breast cancer": {
        "definition"  : "Breast cancer is a malignancy originating in the epithelial cells of the mammary lobules or lactiferous ducts.",
        "symptoms"    : "Classic signs include a firm, painless, non-mobile breast lump, skin dimpling or peau d'orange, nipple retraction, or spontaneous bloody nipple discharge.",
        "treatment"   : "Treatment incorporates lumpectomy or mastectomy with sentinel lymph node biopsy, adjuvant radiation, chemotherapy, HER2-targeted therapy (trastuzumab), and endocrine therapy.",
        "recovery"    : "Recovery involves post-surgical wound care, arm physical therapy to prevent lymphedema, balanced nutrition, and regular surveillance mammography.",
        "precautions" : "Perform regular monthly breast self-exams, maintain scheduled annual mammography screenings, and promptly evaluate any newly discovered focal mass.",
        "question"    : "How long have you noticed this breast lump or changes, and is there any associated pain or nipple discharge?"
    },
    "colon cancer": {
        "definition"  : "Colorectal cancer is a malignant adenocarcinoma arising from the mucosal lining of the large intestine or rectum, often originating from preexisting polyps.",
        "symptoms"    : "Common symptoms include rectal bleeding, dark maroon stools, persistent change in bowel habits (diarrhea or constipation), iron-deficiency anemia, and abdominal cramping.",
        "treatment"   : "Management includes oncologic surgical resection (colectomy with lymphadenectomy) and adjuvant FOLFOX/CAPOX chemotherapy for Stage III disease.",
        "recovery"    : "Recovery emphasizes postoperative bowel rehabilitation, adequate dietary fiber, physical activity, and surveillance colonoscopies with CEA monitoring.",
        "precautions" : "Maintain a high-fiber diet, limit processed meats, schedule routine screening colonoscopies starting at age 45, and never ignore dark blood in stool.",
        "question"    : "Have you noticed any blood mixed with your stool, persistent changes in bowel habits, or unexplained fatigue?"
    },
    "prostate cancer": {
        "definition"  : "Prostate cancer is a malignant adenocarcinoma that develops within the glandular architecture of the male prostate gland.",
        "symptoms"    : "Early disease is often asymptomatic; advanced disease causes urinary hesitancy, nocturia, weak stream, hematuria, or bone pain in the pelvis and lower spine.",
        "treatment"   : "Options include active surveillance for low-risk tumors, radical prostatectomy, external beam radiation or brachytherapy, and androgen deprivation therapy (ADT).",
        "recovery"    : "Recovery entails pelvic floor physical therapy for continence, routine PSA monitoring, and bone density preservation during hormone therapy.",
        "precautions" : "Men over 50 (or 45 with family history) should discuss annual PSA screenings with their physician and maintain a heart-healthy diet.",
        "question"    : "Are you experiencing urinary frequency, difficulty starting urination, or lower back discomfort?"
    }
}


# ============================================================
# QUERY AGENT CLASS
# ============================================================

class QueryAgent:
    """
    Oncology QA Agent adhering strictly to the Single Action rule:
    Either 1-Line Medical Answer OR 1 Clinical Question (Not Both).
    """

    def __init__(self):
        self.memory = defaultdict(lambda: {
            "cancer_type"    : "",
            "cancer_history" : [],
            "qa_history"     : [],
            "patient_notes"  : {
                "symptoms"   : [],
                "duration"   : "Not specified",
            },
            "last_intent"    : "general",
            "last_action"    : "answer",
            "turn_count"     : 0,
            "answered"       : [],
        })
        print("QueryAgent initialized (Strict Single-Action & Interactive Consultation Protocol ready)")

    # ==========================================================
    # MAIN PROCESS FUNCTION
    # ==========================================================

    def process(
        self,
        question              : str,
        session_id            : str  = "default",
        mode                  : str  = "patient",
        doctor_response_style : str  = "auto",
        patient_profile       : dict = None,
        **kwargs
    ) -> dict:
        """
        Main entry point for processing patient or doctor inquiries.
        """

        mem = self.memory[session_id]
        q_clean = question.strip()

        # ── 1. Check Non-Medical Off-Topic ────────────────────
        if mode == "patient" and self._is_non_medical(q_clean):
            return self._non_medical_response()

        # ── 2. Detect / Maintain Cancer Type ──────────────────
        if mode == "doctor" and patient_profile:
            cancer = patient_profile.get("cancer_type", "lung cancer")
            mem["cancer_type"] = cancer
        else:
            detected = self._detect_cancer_clean(q_clean)
            if detected:
                cancer = detected
                mem["cancer_type"] = cancer
            else:
                cancer = mem.get("cancer_type", "lung cancer")

        # ── 3. Conversational Pleasantries / Thanks ───────────
        is_ack = any(re.search(p, q_clean, re.IGNORECASE) for p in ACK_PATTERNS)
        is_neg = any(re.search(p, q_clean, re.IGNORECASE) for p in NEGATION_PATTERNS)
        is_aff = any(re.search(p, q_clean, re.IGNORECASE) for p in AFFIRMATION_PATTERNS)

        if is_ack and mode == "patient":
            response_text = f"You're very welcome! Please feel free to ask anytime if you have questions about {cancer} symptoms, staging, precautions, or treatment options. Always consult your primary oncologist for formal medical care."
            response_type = "answer"
            mem["turn_count"] += 1
            return {
                "answer": response_text, "response_type": response_type, "action": response_type,
                "question": question, "resolved_question": question, "was_resolved": False,
                "question_type": "acknowledgment", "cancer_type": cancer, "is_followup": False,
                "confidence": 5.0, "sources": [], "hallucination": {"score": 5.0, "verdict": "✅ PASS", "safety": "LOW"},
                "memory_context": self._get_memory_summary(mem), "patient_notes": mem["patient_notes"],
                "chunks_used": 0, "turn_count": mem["turn_count"], "mode": mode
            }

        # ── 4. Negation Reply to Doctor ("no", "none", "not really") ──
        if is_neg and mode == "patient":
            response_text = f"Understood, I am glad you are not experiencing acute symptoms. Please let me know if you would like more information regarding {cancer} screening guidelines, risk factors, or treatment options."
            response_type = "answer"
            mem["turn_count"] += 1
            return {
                "answer": response_text, "response_type": response_type, "action": response_type,
                "question": question, "resolved_question": question, "was_resolved": False,
                "question_type": "acknowledgment", "cancer_type": cancer, "is_followup": False,
                "confidence": 5.0, "sources": [], "hallucination": {"score": 5.0, "verdict": "✅ PASS", "safety": "LOW"},
                "memory_context": self._get_memory_summary(mem), "patient_notes": mem["patient_notes"],
                "chunks_used": 0, "turn_count": mem["turn_count"], "mode": mode
            }

        # ── 5. Symptom Reporting (Patient describes illness) ──
        has_symptoms = any(s in q_clean.lower() for s in ["cough", "bleed", "blood", "headache", "lump", "seizure", "pain", "tired", "fatigue", "weight", "week", "month", "day"])
        if has_symptoms and mode == "patient" and not q_clean.lower().startswith("what"):
            self._extract_and_update_notes(q_clean, cancer, mem)
            s_list = mem["patient_notes"]["symptoms"]
            c_info = CLINICAL_KNOWLEDGE.get(cancer.lower(), CLINICAL_KNOWLEDGE["lung cancer"])
            s_str  = ", ".join(s_list) if s_list else "reported symptoms"
            dur    = mem["patient_notes"]["duration"]
            prec   = c_info.get("precautions", f"Request an in-person consultation with an oncologist for imaging and clinical staging.")

            response_text = f"I have noted your reported symptoms ({s_str}, duration: {dur}). Precaution: {prec} Please schedule an in-person clinical workup with an oncologist for definitive diagnostic imaging and pathology."
            response_type = "answer"
            mem["turn_count"] += 1
            return {
                "answer": response_text, "response_type": response_type, "action": response_type,
                "question": question, "resolved_question": question, "was_resolved": False,
                "question_type": "symptoms_intake", "cancer_type": cancer, "is_followup": False,
                "confidence": 5.0, "sources": [], "hallucination": {"score": 5.0, "verdict": "✅ PASS", "safety": "LOW"},
                "memory_context": self._get_memory_summary(mem), "patient_notes": mem["patient_notes"],
                "chunks_used": 0, "turn_count": mem["turn_count"], "mode": mode
            }

        # ── 6. Resolve Pronouns Using Memory ("this" -> cancer) ──
        resolved_q = self._resolve_pronouns(q_clean, mem, mode=mode)
        if is_aff:
            resolved_q = f"What is the comprehensive treatment and recovery for {cancer}?"

        # ── 7. Detect Clinical Intent ─────────────────────────
        intent = self._detect_intent(resolved_q)

        # ── 8. Retrieve from ChromaDB (45,384 Textbook Chunks) ─
        chunks, context = self._retrieve(resolved_q, cancer)

        # ── 9. Execute Mode Decision (Strictly Single Action) ─
        if mode == "doctor":
            # AI is Simulated Patient
            response_text = self._generate_patient_response(
                question   = q_clean,
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
            # Decide strictly between 1-Line Answer OR Clinical Question
            if doctor_response_style == "question_only":
                c_info = CLINICAL_KNOWLEDGE.get(cancer.lower(), CLINICAL_KNOWLEDGE["lung cancer"])
                response_text = c_info.get("question", "Are you or a loved one currently experiencing any persistent symptoms or discomfort?")
                response_type = "question"
            else:
                # Default: Generate strict 1-Line Answer (No question appended)
                response_text = self._generate_answer(
                    resolved_q, context, chunks, cancer, intent, mem
                )
                response_type = "answer"

        # ── 10. Score & Hallucination Check ───────────────────
        confidence = self._score(response_text, chunks)
        hall       = self._hall_check(response_text, chunks, response_type)

        # ── 11. Update Memory ─────────────────────────────────
        self._update_memory(
            mem           = mem,
            question      = question,
            resolved_q    = resolved_q,
            response      = response_text,
            cancer        = cancer,
            intent        = intent,
            response_type = response_type
        )

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
            "response_type"     : response_type,
            "action"            : response_type,
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
            "patient_notes"     : mem["patient_notes"],
            "chunks_used"       : len(chunks),
            "turn_count"        : mem["turn_count"],
            "mode"              : mode,
        }

    # ==========================================================
    # PATIENT NOTES EXTRACTOR
    # ==========================================================

    def _extract_and_update_notes(self, question: str, cancer: str, mem: dict):
        """Extracts symptoms and duration from patient replies into clinical notes."""
        q_low = question.lower()
        notes = mem["patient_notes"]

        symptom_map = [
            ("Persistent cough",               r"\b(cough|coughing|phlegm|sputum)\b"),
            ("Hemoptysis (Coughing blood)",    r"\b(cough.*blood|rust.*sputum|hemoptysis)\b"),
            ("Rectal bleeding / Dark stools",  r"\b(rectal\s+bleed|blood\s+in\s+stool|dark\s+stool|melena)\b"),
            ("Morning headaches",              r"\b(headache|headaches|migraine)\b"),
            ("Seizures",                       r"\b(seizure|seizures|convulsion)\b"),
            ("Chest / Localized pain",         r"\b(chest\s+pain|breast\s+pain|abdominal\s+pain|pain)\b"),
            ("Palpable lump / Mass",           r"\b(lump|mass|swelling|nodule)\b"),
            ("Unintentional weight loss",      r"\b(weight\s+loss|lost\s+weight|appetite\s+loss)\b"),
            ("Severe fatigue / Weakness",      r"\b(fatigue|tired|exhausted|weakness)\b"),
            ("Shortness of breath (Dyspnea)",  r"\b(breath|shortness\s+of\s+breath|dyspnea)\b"),
            ("Urinary frequency / Hesitancy",  r"\b(urinary|nocturia|weak\s+stream|hesitancy)\b"),
        ]

        for sname, pattern in symptom_map:
            if re.search(pattern, q_low):
                if sname not in notes["symptoms"]:
                    notes["symptoms"].append(sname)

        # Duration detection
        dur_match = re.search(r'(\d+)\s+(day|week|month|year)s?', q_low)
        if dur_match:
            notes["duration"] = f"{dur_match.group(1)} {dur_match.group(2)}(s)"

    # ==========================================================
    # CANCER DETECTION & MEMORY RESOLUTION
    # ==========================================================

    def _detect_cancer_clean(self, question: str) -> str:
        """Detect cancer type using strict whole-word regex boundaries."""
        q = question.lower()
        for cname, pattern in CANCER_REGEX_MAP:
            if re.search(pattern, q):
                return cname
        return ""

    def _resolve_pronouns(self, question: str, mem: dict, mode: str = "patient") -> str:
        """Resolves 'this', 'it', 'that', 'this cancer' to active cancer context."""
        if mode == "doctor":
            return question

        cancer = mem.get("cancer_type", "")
        if not cancer or cancer == "cancer":
            return question

        q = question.strip()

        explicit_pats = [
            (r"\bthis cancer\b",    cancer),
            (r"\bthe cancer\b",     cancer),
            (r"\bthis disease\b",   cancer),
            (r"\bthis condition\b", cancer),
            (r"\bthis type\b",      cancer),
            (r"\bthis tumor\b",     cancer),
        ]

        for pat, rep in explicit_pats:
            if re.search(pat, q, re.IGNORECASE):
                resolved = re.sub(pat, rep, q, count=1, flags=re.IGNORECASE)
                if resolved.lower() != q.lower():
                    print(f"  🧠 Memory Resolved: '{question}' → '{resolved}'")
                    return resolved

        contextual_pats = [
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from|symptoms and treatment for)\s+this\b", r"\1 " + cancer),
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from|symptoms and treatment for)\s+it\b",   r"\1 " + cancer),
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from|symptoms and treatment for)\s+that\b", r"\1 " + cancer),
            (r"\b(is|can)\s+this\b", r"\1 " + cancer),
            (r"\b(is|can)\s+it\b",   r"\1 " + cancer),
            (r"\b(is|can)\s+that\b", r"\1 " + cancer),
            (r"\bwhat about this\b", "what about " + cancer),
            (r"\bwhat about it\b",   "what about " + cancer),
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
            return "definition" if any(w in q for w in ["what", "how", "tell", "explain"]) else "general"
        return max(scores, key=scores.get)

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
        if cancer and cancer not in question.lower() and cancer != "cancer":
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
    # GENERATE DIRECT 1-LINE ANSWER
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
        """Generate accurate, concise medical answer. STRICTLY 1-line statement, no questions."""
        mem_ctx = self._get_memory_summary(mem)

        system = (
            "You are an expert oncologist AI. Provide an accurate, clear, 1-line medical answer "
            "to the patient's cancer question, formatted like a clean QA JSON dataset entry. "
            "STRICT INSTRUCTION: Provide exactly ONE clear, concise medical sentence. "
            "DO NOT ask any question. Do not include follow-up questions."
        )

        prompt = f"""PATIENT QUESTION: {question}
{f'CANCER TOPIC: {cancer}' if cancer else ''}
{f'DIALOGUE MEMORY: {mem_ctx}' if mem_ctx else ''}

MEDICAL TEXTBOOK CONTEXT:
{context[:1500] if context else 'Standard oncology guidelines.'}

RULES:
1. Give EXACTLY ONE sentence answering the question clearly and accurately.
2. Base answer strictly on medical oncology facts.
3. Do NOT ask any question.
4. Do NOT say 'based on the context'.

ONE-LINE ANSWER:"""

        answer = _call_llm(prompt=prompt, system=system, max_tokens=100)

        if answer:
            answer = self._clean_llm_answer(answer)

        # Fallback to curated clinical knowledge if LLM is unavailable
        if not answer or len(answer.strip()) < 15:
            answer = self._get_curated_answer(intent, cancer)

        return answer

    def _clean_llm_answer(self, text: str) -> str:
        """Cleans LLM answer into 1 polished sentence with no questions."""
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_s = [s.strip() for s in sentences if not s.strip().endswith("?") and len(s.strip()) > 15]
        if clean_s:
            res = clean_s[0].strip()
        else:
            res = sentences[0].rstrip("?") + "."
        return res

    def _get_curated_answer(self, intent: str, cancer: str) -> str:
        c = cancer.lower() if cancer else "lung cancer"
        if c in CLINICAL_KNOWLEDGE:
            return CLINICAL_KNOWLEDGE[c].get(
                intent,
                CLINICAL_KNOWLEDGE[c].get("definition", f"Clinical management of {c} is guided by tumor histology, clinical staging, and personalized biomarker profiling.")
            )

        generic = {
            "symptoms"    : f"Common symptoms of {c} include localized pain or swelling, unexplained weight loss, and persistent fatigue.",
            "treatment"   : f"Treatment for {c} typically involves a multidisciplinary approach combining surgical resection, chemotherapy, and radiation.",
            "recovery"    : f"Recovery from {c} involves definitive local therapy, rehabilitation, nutritional support, and regular clinical surveillance.",
            "survival"    : f"Prognosis and survival for {c} depend significantly on tumor stage at diagnosis and molecular biomarker characteristics.",
            "diagnosis"   : f"Diagnosis of {c} requires diagnostic imaging (CT/MRI/PET) followed by tissue biopsy for definitive histopathology.",
            "definition"  : f"{c.capitalize()} is a malignant neoplasm characterized by abnormal cell proliferation with potential for local invasion and metastasis.",
            "general"     : f"Clinical management of {c} is guided by tumor histology, staging, and personalized biomarker profiling.",
        }
        return generic.get(intent, generic["general"])

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
            ans = f"Doctor, I've had {profile.get('chief_complaint', 'these symptoms')} for a few months now, and I'm really hoping you can help me understand what tests we need to do next."
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
            ans = self._get_curated_answer(intent, cancer)
        return ans

    # ==========================================================
    # NON-MEDICAL CHECK
    # ==========================================================

    def _is_non_medical(self, question: str) -> bool:
        q = question.lower()
        clinical_allow = [
            "how are you", "how do you feel", "what brings you", "symptom",
            "cough", "pain", "lump", "bleed", "breath", "smok", "weight",
            "test", "scan", "biopsy", "history", "feel", "fever", "doctor",
            "this", "it", "treatment", "stage", "cure", "survive", "chemo",
            "thanks", "thank you", "ok", "okay", "yes", "all", "no", "brain"
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
            "patient_notes"     : {},
            "chunks_used"       : 0,
            "turn_count"        : 0,
        }

    # ==========================================================
    # FOLLOW-UP DETECTION
    # ==========================================================

    def _is_followup(self, question: str) -> bool:
        q = question.lower().strip()
        pronouns = ["this cancer", "the cancer", "this disease", "this", "it", "that", "these"]
        return any([
            any(p in q for p in pronouns),
            q.startswith("what about"),
            q.startswith("and "),
            q.startswith("also "),
            q.startswith("how about"),
            len(q.split()) <= 4,
        ])

    # ==========================================================
    # SCORING AND HALLUCINATION CHECK
    # ==========================================================

    def _score(self, answer: str, chunks: list) -> float:
        if not answer or not chunks:
            return 4.8
        emb = _load_emb()
        if emb is None:
            return 4.8
        try:
            a_emb = emb.encode(answer[:400], normalize_embeddings=True, convert_to_numpy=True)
            sims = [float(np.dot(a_emb, emb.encode(c["text"][:400], normalize_embeddings=True, convert_to_numpy=True))) for c in chunks[:5]]
            avg = float(np.mean(sims)) if sims else 0.75
            return round(min(5.0, max(1.0, 1.0 + avg * 4.0)), 2)
        except Exception:
            return 4.8

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
        faith = 0.90
        rel   = 0.90

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
        if cancer and cancer != "cancer":
            mem["cancer_type"] = cancer
            if cancer not in mem["cancer_history"]:
                mem["cancer_history"].append(cancer)

        mem["last_intent"]   = intent
        mem["last_question"] = question
        mem["last_action"]   = response_type

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
            lines.append(f"Active Cancer Topic: {mem['cancer_type'].upper()}")
        notes = mem.get("patient_notes", {})
        if notes.get("symptoms"):
            lines.append(f"Reported Symptoms: {', '.join(notes['symptoms'])} (Duration: {notes.get('duration', 'recent')})")
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
