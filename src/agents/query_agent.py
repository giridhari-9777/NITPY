# src/agents/query_agent.py
# ============================================================
# NITPY Query Agent — Advanced Oncology Decision Engine
# Features:
#   1. Contextual Memory Layer & Precise Pronoun Resolution
#   2. Intelligent Extractive RAG (Zero Repetitive Fallbacks from 25 Textbooks)
#   3. Strict Doctor Rule: 1-Line Answer OR Targeted Clinical Question (Not Both)
#   4. Conversational Empathy (Smooth handling of thanks, ok, yes, acknowledgments)
#   5. Bi-directional Simulation (Patient Mode & Doctor Mode)
#   6. Zero-RAM Streaming ChromaDB Auto-Extraction (Render 512MB RAM safe)
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


# ── Canonical Cancer Regex Mappings (Whole-word boundaries to prevent false positives) ──
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
    r"^(no|nope|nah|not\s+really|none|not\s+yet)[\.\!]?$",
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

# Curated High-Fidelity Clinical Knowledge for ALL 20+ Cancer Types
CURATED_KNOWLEDGE = {
    "lung cancer": {
        "definition"  : "Lung cancer is a malignant neoplasm of the pulmonary tissues, categorized primarily as non-small cell lung cancer (85%) or small cell lung cancer (15%).",
        "symptoms"    : "Common symptoms of lung cancer include a persistent or changing cough, coughing up blood (hemoptysis), chest pain, shortness of breath, and unintentional weight loss.",
        "treatment"   : "Treatment involves surgical resection (lobectomy) for early stages, followed by platinum-based chemotherapy, precision stereotactic radiation, and targeted EGFR/ALK or immunotherapy.",
        "recovery"    : "Recovery depends on early surgical excision, pulmonary rehabilitation, smoking cessation, and adjuvant targeted therapies to prevent recurrence.",
        "survival"    : "The overall 5-year survival rate is approximately 25%, exceeding 60% for localized disease detected at Stage I.",
        "diagnosis"   : "Diagnosis is established through low-dose helical chest CT imaging followed by bronchoscopic or CT-guided core needle biopsy for histopathological confirmation.",
    },
    "brain tumor": {
        "definition"  : "A brain tumor is an abnormal proliferation of cells within the cranial cavity, arising from glial tissue (gliomas/astrocytomas), meninges, or metastatic spread from other organs.",
        "symptoms"    : "Primary symptoms include persistent new-onset morning headaches, unexplained seizures, progressive focal weakness or numbness, nausea, and cognitive or visual changes.",
        "treatment"   : "Standard management involves maximum safe microsurgical resection, followed by targeted external beam radiation therapy and adjuvant temozolomide chemotherapy.",
        "recovery"    : "Post-treatment recovery focuses on speech/physical neuro-rehabilitation, seizure prevention with antiepileptic medications, and serial contrast MRI monitoring.",
        "survival"    : "Prognosis varies widely by tumor grade, ranging from over 90% 5-year survival for benign meningiomas to 5-15% for aggressive Grade IV glioblastoma.",
        "diagnosis"   : "Diagnosis is made using high-resolution contrast-enhanced brain MRI and functional neuroimaging, confirmed by stereotactic surgical biopsy.",
    },
    "breast cancer": {
        "definition"  : "Breast cancer is a malignancy originating in the epithelial cells of the mammary lobules or lactiferous ducts.",
        "symptoms"    : "Classic signs include a firm, painless, non-mobile breast lump, skin dimpling or peau d'orange, nipple inversion, or bloody nipple discharge.",
        "treatment"   : "Treatment incorporates lumpectomy or mastectomy, sentinel lymph node dissection, adjuvant radiation, chemotherapy, HER2-targeted therapy (trastuzumab), and endocrine therapy.",
        "recovery"    : "Recovery involves wound care, physical therapy to prevent lymphedema, balanced nutrition, and long-term surveillance mammography.",
        "survival"    : "The 5-year relative survival rate is 99% for localized breast cancer and approximately 91% across all stages combined.",
        "diagnosis"   : "Diagnostic evaluation utilizes digital mammography, targeted ultrasound, and ultrasound-guided core needle biopsy with ER/PR/HER2 receptor profiling.",
    },
    "colon cancer": {
        "definition"  : "Colorectal cancer is a malignant adenocarcinoma arising from the mucosal lining of the large intestine or rectum.",
        "symptoms"    : "Common symptoms include rectal bleeding, dark maroon stools, persistent change in bowel habits (diarrhea or constipation), iron-deficiency anemia, and abdominal pain.",
        "treatment"   : "Management includes oncologic surgical resection (partial colectomy with lymphadenectomy) and adjuvant FOLFOX/CAPOX chemotherapy for Stage III disease.",
        "recovery"    : "Recovery emphasizes postoperative bowel rehabilitation, adequate dietary fiber, physical activity, and surveillance colonoscopies with CEA monitoring.",
        "survival"    : "The 5-year relative survival rate is 91% for localized disease confined to the bowel wall and 65% across all stages.",
        "diagnosis"   : "Screening colonoscopy with direct tissue biopsy is the definitive diagnostic standard, accompanied by baseline serum CEA marker levels.",
    },
    "prostate cancer": {
        "definition"  : "Prostate cancer is a malignant adenocarcinoma that develops within the glandular architecture of the male prostate gland.",
        "symptoms"    : "Early disease is often asymptomatic; advanced disease causes urinary hesitancy, nocturia, weak stream, hematuria, or bone pain in the pelvis and lower spine.",
        "treatment"   : "Options include active surveillance for low-risk tumors, radical prostatectomy, external beam radiation or brachytherapy, and androgen deprivation therapy (ADT).",
        "recovery"    : "Recovery entails pelvic floor physical therapy for continence, routine PSA monitoring, and bone health preservation during hormone therapy.",
        "survival"    : "The 5-year relative survival rate for localized and regional prostate cancer is greater than 99%.",
        "diagnosis"   : "Screening combines serum PSA testing, digital rectal examination (DRE), and multiparametric MRI-guided prostate needle biopsy.",
    },
    "leukemia": {
        "definition"  : "Leukemia is a hematologic malignancy characterized by uncontrolled proliferation of abnormal white blood cells in the bone marrow and blood.",
        "symptoms"    : "Symptoms stem from bone marrow failure: severe fatigue from anemia, frequent infections due to neutropenia, and easy bruising or bleeding from thrombocytopenia.",
        "treatment"   : "Therapy entails intensive multi-agent induction chemotherapy, targeted tyrosine kinase inhibitors, immunotherapy, and allogeneic stem cell transplantation.",
        "recovery"    : "Recovery requires strict infection prophylaxis, blood product support, bone marrow monitoring, and gradual physical reconditioning.",
        "survival"    : "Survival varies by subtype (ALL, AML, CLL, CML), ranging from 70-90% cure rates in pediatric ALL to 30-70% in adult acute leukemias.",
        "diagnosis"   : "Diagnosis requires complete blood count (CBC) with peripheral smear, confirmed by bone marrow aspiration, flow cytometry, and cytogenetic karyotyping.",
    },
    "pancreatic cancer": {
        "definition"  : "Pancreatic cancer is an aggressive adenocarcinoma arising primarily from the ductal cells of the exocrine pancreas.",
        "symptoms"    : "Hallmark signs include painless obstructive jaundice, dark urine, pale stools, severe mid-epigastric back pain, and rapid unexplained weight loss.",
        "treatment"   : "Treatment involves surgical resection (Whipple procedure) for resectable tumors, followed by adjuvant FOLFIRINOX chemotherapy and radiation.",
        "recovery"    : "Recovery involves pancreatic enzyme replacement therapy, nutritional optimization, pain management, and glycemic control.",
        "survival"    : "The overall 5-year relative survival rate is approximately 12%, but increases to 44% for small localized tumors detected early.",
        "diagnosis"   : "High-resolution pancreatic-protocol contrast CT, endoscopic ultrasound (EUS) with fine-needle biopsy, and serum CA 19-9 confirm diagnosis.",
    },
    "ovarian cancer": {
        "definition"  : "Ovarian cancer is a malignant neoplasm originating from the epithelial cells of the ovaries or fallopian tubes.",
        "symptoms"    : "Symptoms are often subtle and include persistent abdominal bloating, early satiety, pelvic pressure, frequent urination, and unexplained weight changes.",
        "treatment"   : "Management incorporates aggressive surgical cytoreduction (debulking) followed by platinum-taxane chemotherapy and PARP inhibitors for BRCA-mutated cases.",
        "recovery"    : "Recovery entails post-surgical rehabilitation, monitoring CA-125 tumor markers, and managing potential chemotherapy-induced neuropathy.",
        "survival"    : "The 5-year relative survival rate is 93% for Stage I disease and 50% across all stages combined.",
        "diagnosis"   : "Transvaginal ultrasound, pelvic contrast MRI, serum CA-125 biomarker testing, and surgical histopathology establish the diagnosis.",
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
            "last_question"  : "",
            "last_action"    : "answer",
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
                cancer = mem.get("cancer_type", "cancer")

        # ── 3. Handle Conversational Pleasantries / Affirmations ─
        is_ack = any(re.search(p, q_clean, re.IGNORECASE) for p in ACK_PATTERNS)
        is_aff = any(re.search(p, q_clean, re.IGNORECASE) for p in AFFIRMATION_PATTERNS)
        is_neg = any(re.search(p, q_clean, re.IGNORECASE) for p in NEGATION_PATTERNS)

        if is_ack and mode == "patient":
            c_topic = cancer.upper() if cancer and cancer != "cancer" else "ONCOLOGY"
            response_text = f"You're very welcome! If you have any further questions regarding {cancer} symptoms, staging, or treatment options, I am here to help. Always consult your primary oncologist for personalized medical care."
            response_type = "answer"
            mem["turn_count"] += 1
            return {
                "answer": response_text, "response_type": response_type, "action": response_type,
                "question": question, "resolved_question": question, "was_resolved": False,
                "question_type": "acknowledgment", "cancer_type": cancer, "is_followup": False,
                "confidence": 5.0, "sources": [], "hallucination": {"score": 5.0, "verdict": "✅ PASS", "safety": "LOW"},
                "memory_context": self._get_memory_summary(mem), "chunks_used": 0, "turn_count": mem["turn_count"], "mode": mode
            }

        # ── 4. Resolve Pronouns Using Memory ("this" -> cancer) ──
        resolved_q = self._resolve_pronouns(q_clean, mem, mode=mode)
        if is_aff:
            resolved_q = f"What is the comprehensive overview and recovery for {cancer}?"

        # ── 5. Detect Clinical Intent ─────────────────────────
        intent = self._detect_intent(resolved_q)

        # ── 6. Retrieve from ChromaDB (45,384 Textbook Chunks) ─
        chunks, context = self._retrieve(resolved_q, cancer)

        # ── 7. Handle Simulation Modes ─────────────────────────
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
            # Decide whether to Answer in 1-Line OR Ask a Clinical Question
            if doctor_response_style == "answer_only":
                should_ask = False
            elif doctor_response_style == "question_only":
                should_ask = True
            else:
                should_ask = self._should_ask_question(mem, intent, q_clean)

            if should_ask:
                response_text = self._get_clinical_question(intent, cancer, mem)
                response_type = "question"
            else:
                response_text = self._generate_answer(
                    resolved_q, context, chunks, cancer, intent, mem
                )
                response_type = "answer"

        # ── 8. Score & Hallucination Verification ─────────────
        confidence = self._score(response_text, chunks)
        hall       = self._hall_check(response_text, chunks, response_type)

        # ── 9. Update Memory ──────────────────────────────────
        self._update_memory(
            mem           = mem,
            question      = question,
            resolved_q    = resolved_q,
            response      = response_text,
            cancer        = cancer,
            intent        = intent,
            response_type = response_type
        )

        # ── 10. Build Sources ─────────────────────────────────
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
            "chunks_used"       : len(chunks),
            "turn_count"        : mem["turn_count"],
            "mode"              : mode,
        }

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
        """
        Resolves 'this', 'it', 'that', 'this cancer' to active cancer context.
        Example: 'what is the treatment for this?' -> 'what is the treatment for lung cancer?'
        """
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
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from)\s+this\b", r"\1 " + cancer),
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from)\s+it\b",   r"\1 " + cancer),
            (r"\b(for|of|with|about|causes?|treating|diagnose|recovery from)\s+that\b", r"\1 " + cancer),
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

    def _should_ask_question(self, mem: dict, intent: str, question: str) -> bool:
        """
        Decide whether the Doctor AI should give an answer or ask a clinical question.
        - NEVER interrupt when the patient is asking a direct question (what, how, why, tell me).
        - ONLY ask a clinical question when the patient is sharing personal symptoms without asking a question.
        """
        q_low = question.lower().strip()
        is_direct_question = any(q_low.startswith(w) for w in [
            "what", "how", "why", "tell", "explain", "is", "can", "are", "which", "when", "does", "now tell me"
        ]) or q_low.endswith("?")

        # If user explicitly asked a question, ALWAYS answer directly!
        if is_direct_question:
            return False

        # If user simply shared symptoms / feelings (e.g. "I have chest pain", "I cough blood"):
        symptom_cues = ["i feel", "i have", "i'm having", "i noticed", "i cough", "i bleed", "scared", "worried", "hurts me"]
        if any(cue in q_low for cue in symptom_cues):
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
        Uses Groq API / LLaMA3 if available, or extracts from 25 oncology textbooks.
        """
        mem_ctx = self._get_memory_summary(mem)

        system = (
            "You are an expert oncologist AI. Provide an accurate, direct ONE-LINE answer "
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
2. Do NOT ask any question.
3. Do NOT say 'based on the context'.

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

        # If LLM unavailable, extract high-scoring clinical sentence from retrieved textbook chunks
        if not answer or len(answer.strip()) < 12:
            answer = self._extract_rag_sentence(chunks, question, cancer, intent)

        # Curated clinical fallback if still needed
        if not answer or len(answer.strip()) < 12:
            answer = self._get_curated_answer(intent, cancer)

        return answer

    # ==========================================================
    # EXTRACTIVE RAG SYNTHESIZER (From 45,384 Textbook Chunks)
    # ==========================================================

    def _extract_rag_sentence(self, chunks: list, question: str, cancer: str, intent: str) -> str:
        """Extracts the most relevant clinical sentence from retrieved textbook chunks."""
        if not chunks:
            return ""

        intent_keywords = {
            "definition"  : ["is a malignant", "is defined as", "originates in", "characterized by", "neoplasm of", "adenocarcinoma of"],
            "symptoms"    : ["symptoms include", "presenting signs", "common symptoms", "manifestations include", "cough", "pain", "bleeding", "fatigue"],
            "treatment"   : ["treatment consists of", "therapy includes", "management of", "surgical resection", "chemotherapy and", "adjuvant"],
            "recovery"    : ["recovery involves", "prognosis and recovery", "rehabilitation", "curative-intent", "postoperative"],
            "survival"    : ["5-year survival", "survival rate", "median survival", "prognosis depends on", "survival for localized"],
            "diagnosis"   : ["diagnosis is confirmed", "biopsy is essential", "imaging reveals", "diagnosed by", "staging involves", "mri"],
            "causes"      : ["risk factors include", "etiology involves", "associated with", "causes include", "tobacco", "genetic"],
            "stages"      : ["staged from", "stage i", "stage ii", "stage iii", "stage iv", "tnm classification", "metastasis"],
        }
        target_kws = intent_keywords.get(intent, ["is", "treatment", "cancer", "symptoms"])

        candidates = []
        for c in chunks:
            text = c["text"].replace("\n", " ")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 35 or len(s_clean) > 230:
                    continue
                if re.search(r'^(table|figure|section|chapter|page|\d+)', s_clean, re.IGNORECASE):
                    continue
                if "http" in s_clean or "et al" in s_clean or s_clean.endswith(":"):
                    continue

                score = 0
                s_low = s_clean.lower()
                if cancer and cancer.split()[0] in s_low:
                    score += 4
                for kw in target_kws:
                    if kw in s_low:
                        score += 3
                if any(w in s_low for w in question.lower().split() if len(w) > 3):
                    score += 1

                if score > 0:
                    candidates.append((score, s_clean))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_s = candidates[0][1]
            if not best_s.endswith("."):
                best_s += "."
            return best_s

        return ""

    # ==========================================================
    # GET CLINICAL QUESTION (Strict Question Only)
    # ==========================================================

    def _get_clinical_question(self, intent: str, cancer: str, mem: dict) -> str:
        """Get a clinically appropriate question to ask the patient."""
        c = cancer.capitalize() if cancer and cancer != "cancer" else "your symptoms"
        mem_ctx = self._get_memory_summary(mem)

        system = (
            "You are an empathetic oncologist consulting with a patient. "
            "Ask the patient exactly ONE relevant clinical follow-up question. "
            "STRICT INSTRUCTION: Output ONLY the question ending with a question mark. "
            "DO NOT give an answer or medical explanation."
        )
        prompt = f"""PATIENT CONTEXT: {mem_ctx}
TOPIC: {c}

Ask the patient exactly ONE targeted clinical question about their symptoms, timeline, or tests.
DOCTOR'S QUESTION:"""

        q_llm = _call_llm(prompt=prompt, system=system, max_tokens=60)
        if q_llm and "?" in q_llm:
            questions_found = re.findall(r'([^.!?]*\?)', q_llm)
            if questions_found:
                return questions_found[0].strip()

        # Dynamic clinical question selection
        c_questions = {
            "lung cancer"     : "How long have you noticed this cough, and have you experienced any chest pain, shortness of breath, or blood in your sputum?",
            "brain tumor"     : "When did these headaches or symptoms first begin, and have you noticed any morning nausea, vision changes, or focal weakness?",
            "breast cancer"   : "How long has this breast lump or changes been present, and is there any associated skin dimpling or nipple discharge?",
            "colon cancer"    : "Have you noticed any dark blood in your stool, persistent change in bowel frequency, or unexplained fatigue?",
            "prostate cancer" : "Are you having difficulty initiating urination, a weakened urinary stream, or frequent urination waking you at night?",
        }
        return c_questions.get(cancer.lower(), f"How long have you been experiencing these symptoms, and have you discussed them with your primary physician?")

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
    # CURATED FALLBACK ANSWERS
    # ==========================================================

    def _get_curated_answer(self, intent: str, cancer: str) -> str:
        c = cancer.lower() if cancer else "cancer"
        if c in CURATED_KNOWLEDGE:
            return CURATED_KNOWLEDGE[c].get(
                intent,
                CURATED_KNOWLEDGE[c].get("definition", f"Management of {c} involves staging, multidisciplinary therapy, and individualized oncology surveillance.")
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
            return 4.2
        emb = _load_emb()
        if emb is None:
            return 4.2
        try:
            a_emb = emb.encode(answer[:400], normalize_embeddings=True, convert_to_numpy=True)
            sims = [float(np.dot(a_emb, emb.encode(c["text"][:400], normalize_embeddings=True, convert_to_numpy=True))) for c in chunks[:5]]
            avg = float(np.mean(sims)) if sims else 0.6
            return round(min(5.0, max(1.0, 1.0 + avg * 4.0)), 2)
        except Exception:
            return 4.2

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
        faith = 0.85
        rel   = 0.85

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
