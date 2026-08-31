# src/generator/generator_model.py

import os
import sys
import pickle
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# CONFIG
# ==================================================

EMBEDDINGS_FOLDER = "outputs/embeddings"
OLLAMA_URL        = "http://localhost:11434/api/generate"
MODEL_NAME        = "llama3"
TOP_K             = 5


# ==================================================
# QUESTION TYPE CLASSIFIER
# ==================================================

QUESTION_TYPES = {
    "treatment"  : [
        "treat", "treatment", "therapy", "chemo",
        "chemotherapy", "radiation", "surgery", "drug",
        "medicine", "medication", "cure", "manage",
        "dose", "regimen", "protocol", "operation"
    ],
    "diagnosis"  : [
        "diagnose", "diagnosis", "detect", "test",
        "biopsy", "scan", "mri", "ct", "xray",
        "x-ray", "blood test", "confirm", "identify",
        "check", "screening", "examine"
    ],
    "symptoms"   : [
        "symptom", "sign", "feel", "pain", "cough",
        "tired", "fatigue", "weight loss", "bleeding",
        "lump", "swelling", "nausea", "vomiting",
        "fever", "ache", "discomfort", "notice"
    ],
    "prognosis"  : [
        "survival", "survive", "prognosis",
        "life expectancy", "how long", "outlook",
        "chance", "rate", "percentage", "recover",
        "remission", "recurrence", "spread"
    ],
    "radiology"  : [
        "imaging", "radiology", "pet scan", "mri",
        "ct scan", "ultrasound", "x-ray", "mammogram",
        "nuclear", "contrast", "image", "scan"
    ],
    "pathology"  : [
        "pathology", "biopsy", "histology", "cell",
        "tissue", "grade", "marker", "her2", "er",
        "pr", "mutation", "genetic", "molecular",
        "biomarker", "receptor"
    ],
    "staging"    : [
        "stage", "staging", "tnm", "metastasis",
        "spread", "lymph node", "extent", "localized",
        "advanced", "early stage", "late stage",
        "grade"
    ],
    "prevention" : [
        "prevent", "prevention", "risk", "avoid",
        "reduce", "screening", "vaccine", "lifestyle",
        "diet", "exercise", "smoking", "alcohol",
        "hereditary", "genetic risk", "family history"
    ]
}


def classify_question(question: str) -> str:

    question_lower = question.lower()
    scores         = {}

    for qtype, keywords in QUESTION_TYPES.items():
        scores[qtype] = sum(
            1 for kw in keywords
            if kw in question_lower
        )

    best_type = max(scores, key=lambda x: scores[x])
    return best_type if scores[best_type] > 0 else "general"


# ==================================================
# CANCER TYPE EXTRACTOR
# ==================================================

def extract_cancer_type(question: str) -> str:

    cancer_types = [
        "lung cancer", "breast cancer", "colon cancer",
        "colorectal cancer", "prostate cancer",
        "leukemia", "lymphoma", "melanoma",
        "pancreatic cancer", "liver cancer",
        "cervical cancer", "ovarian cancer",
        "stomach cancer", "bone cancer",
        "brain cancer", "bladder cancer",
        "kidney cancer", "thyroid cancer",
        "esophageal cancer", "head and neck cancer"
    ]

    question_lower = question.lower()

    for cancer in cancer_types:
        if cancer in question_lower:
            return cancer

    if "cancer" in question_lower:
        words = question_lower.split()
        for i, word in enumerate(words):
            if word == "cancer" and i > 0:
                return f"{words[i-1]} cancer"

    return "cancer"


# ==================================================
# SUB QUERY GENERATOR
# ==================================================

SUB_QUERY_TEMPLATES = {
    "treatment"  : [
        "What are the standard treatment options for {cancer}?",
        "What is the first-line chemotherapy regimen for {cancer}?",
        "How is surgery used in treating {cancer}?",
        "What are the latest targeted therapies for {cancer}?",
        "What are the side effects of {cancer} treatment?"
    ],
    "diagnosis"  : [
        "How is {cancer} diagnosed clinically?",
        "What blood tests are used to detect {cancer}?",
        "What imaging techniques confirm {cancer} diagnosis?",
        "What are the diagnostic criteria for {cancer}?",
        "How is {cancer} confirmed through biopsy?"
    ],
    "symptoms"   : [
        "What are the early warning signs of {cancer}?",
        "What are the most common symptoms of {cancer}?",
        "How do {cancer} symptoms differ by stage?",
        "What physical changes indicate {cancer}?",
        "When should a patient seek advice for {cancer} symptoms?"
    ],
    "prognosis"  : [
        "What is the 5-year survival rate for {cancer}?",
        "How does stage affect {cancer} prognosis?",
        "What factors influence {cancer} survival rates?",
        "What is the recurrence rate after {cancer} treatment?",
        "How is remission defined in {cancer} patients?"
    ],
    "radiology"  : [
        "What imaging modality is best for {cancer}?",
        "How is PET scan used in {cancer} staging?",
        "What does {cancer} look like on CT scan?",
        "How often should {cancer} patients get imaging?",
        "What are radiological features of {cancer} metastasis?"
    ],
    "pathology"  : [
        "What are the histological subtypes of {cancer}?",
        "What biomarkers are tested in {cancer} pathology?",
        "How is {cancer} graded pathologically?",
        "What genetic mutations are linked to {cancer}?",
        "How does molecular profiling guide {cancer} treatment?"
    ],
    "staging"    : [
        "What is the TNM staging system for {cancer}?",
        "What defines stage 1 vs stage 4 {cancer}?",
        "How is lymph node involvement assessed in {cancer}?",
        "What staging investigations are done for {cancer}?",
        "How does staging affect {cancer} treatment decisions?"
    ],
    "prevention" : [
        "What are the major risk factors for {cancer}?",
        "How can {cancer} be prevented effectively?",
        "What lifestyle changes reduce {cancer} risk?",
        "Is {cancer} hereditary and how to manage risk?",
        "What screening programs exist for {cancer} detection?"
    ],
    "general"    : [
        "What is {cancer} and how does it develop?",
        "What are the main types of {cancer}?",
        "How common is {cancer} globally?",
        "What are current research directions for {cancer}?",
        "What support is available for {cancer} patients?"
    ]
}


def generate_sub_queries(
    question      : str,
    question_type : str
) -> list:

    cancer_type = extract_cancer_type(question)
    templates   = SUB_QUERY_TEMPLATES.get(
        question_type,
        SUB_QUERY_TEMPLATES["general"]
    )

    return [
        t.format(cancer=cancer_type)
        for t in templates[:5]
    ]


# ==================================================
# LOAD EMBEDDINGS + CHUNKS
# ==================================================

def load_embeddings_and_chunks():

    meta_path = os.path.join(
        EMBEDDINGS_FOLDER, "chunks_metadata.pkl"
    )
    emb_path  = os.path.join(
        EMBEDDINGS_FOLDER, "embeddings.npy"
    )

    if not os.path.exists(meta_path):
        print(f"ERROR: chunks_metadata.pkl not found!")
        print(f"Run: python3 src/embeddings/embeddings_pipeline.py")
        sys.exit(1)

    if not os.path.exists(emb_path):
        print(f"ERROR: embeddings.npy not found!")
        print(f"Run: python3 src/embeddings/embeddings_pipeline.py")
        sys.exit(1)

    print("Loading chunks and embeddings...")

    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.load(emb_path)

    print(f"  Chunks     : {len(chunks)}")
    print(f"  Embeddings : {embeddings.shape}")
    print("  Loaded ✅\n")

    return chunks, embeddings


# ==================================================
# RETRIEVE TOP K CHUNKS
# ==================================================

def retrieve_chunks(
    question   : str,
    chunks     : list,
    embeddings : np.ndarray,
    model,
    top_k      : int = TOP_K
) -> list:

    # Encode question
    q_emb = model.encode(
        question,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    # Cosine similarity
    scores  = np.dot(embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        chunk = chunks[idx].copy()
        chunk["score"] = round(float(scores[idx]), 4)
        results.append(chunk)

    return results


# ==================================================
# FORMAT CONTEXT
# ==================================================

def format_context(chunks: list) -> str:

    parts = []

    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Reference {i+1}]\n"
            f"Source : {chunk.get('source', 'unknown')}\n"
            f"Title  : {chunk.get('title',  'no title')}\n"
            f"Score  : {chunk.get('score',  0):.4f}\n\n"
            f"{chunk['text']}"
        )

    return "\n\n" + "─"*50 + "\n\n".join(parts)


# ==================================================
# CALL OLLAMA
# ==================================================

def call_ollama(prompt: str) -> str:

    try:
        resp = requests.post(
            OLLAMA_URL,
            json    = {
                "model"  : MODEL_NAME,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : 0.3,
                    "num_predict" : 800
                }
            },
            timeout = 120
        )

        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return ""

    except requests.exceptions.ConnectionError:
        return ""

    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


# ==================================================
# BUILD DOCTOR PROMPT
# ==================================================

def build_prompt(
    question      : str,
    context       : str,
    question_type : str,
    sub_queries   : list,
    cancer_type   : str
) -> str:

    sub_q = "\n".join(
        f"  {i+1}. {q}"
        for i, q in enumerate(sub_queries)
    )

    return f"""You are an expert oncologist AI assistant
in a Doctor-Patient simulation chatbot.

STRICT RULES:
- Answer ONLY from the provided medical context below
- Be empathetic, clear and compassionate
- Use phrases like: "typically", "may", "generally",
  "research suggests", "studies show"
- Structure answer with numbered points
- End with: "Please consult your oncologist for
  personalized medical advice."
- NEVER make up facts not in the context

MEDICAL CONTEXT FROM ONCOLOGY TEXTBOOKS:
{context}

PATIENT QUESTION   : {question}
QUESTION TYPE      : {question_type.upper()}
CANCER TYPE        : {cancer_type}

RELATED SUB-QUERIES FOR CONTEXT:
{sub_q}

Doctor's Response:"""


# ==================================================
# FALLBACK ANSWER
# ==================================================

def fallback_answer(context: str) -> str:

    sentences = context.replace("\n", " ").split(". ")
    relevant  = ". ".join(sentences[:4])

    return (
        f"Based on medical literature: {relevant}. "
        f"Research suggests individual results may vary. "
        f"Please consult your oncologist for "
        f"personalized medical advice."
    )


# ==================================================
# MAIN ANSWER FUNCTION
# ==================================================

def answer_question(
    question   : str,
    chunks     : list,
    embeddings : np.ndarray,
    model
) -> dict:

    # Step 1 — Classify
    question_type = classify_question(question)
    sub_queries   = generate_sub_queries(
        question, question_type
    )
    cancer_type   = extract_cancer_type(question)

    # Step 2 — Retrieve top chunks
    top_chunks = retrieve_chunks(
        question   = question,
        chunks     = chunks,
        embeddings = embeddings,
        model      = model,
        top_k      = TOP_K
    )

    # Step 3 — Format context
    context = format_context(top_chunks)

    # Step 4 — Build prompt
    prompt = build_prompt(
        question      = question,
        context       = context,
        question_type = question_type,
        sub_queries   = sub_queries,
        cancer_type   = cancer_type
    )

    # Step 5 — Generate
    answer = call_ollama(prompt)

    if not answer:
        answer = fallback_answer(context)

    return {
        "question"      : question,
        "answer"        : answer,
        "question_type" : question_type,
        "cancer_type"   : cancer_type,
        "sub_queries"   : sub_queries,
        "sources"       : top_chunks
    }


# ==================================================
# PRINT RESPONSE
# ==================================================

def print_response(result: dict):

    print(f"\n{'='*60}")
    print(f"  DOCTOR-PATIENT CONSULTATION")
    print(f"{'='*60}")
    print(f"  Question Type : {result['question_type'].upper()}")
    print(f"  Cancer Type   : {result['cancer_type']}")

    print(f"\n  Related Sub-Queries:")
    for i, sq in enumerate(result["sub_queries"]):
        print(f"     {i+1}. {sq}")

    print(f"\n  Sources Retrieved:")
    for i, src in enumerate(result["sources"][:3]):
        print(
            f"     {i+1}. {src.get('source', 'unknown')}"
            f" (score: {src.get('score', 0):.4f})"
        )

    print(f"\n  Doctor's Answer:")
    print(f"  {'-'*56}")
    for line in result["answer"].split("\n"):
        if line.strip():
            print(f"  {line}")
    print(f"{'='*60}\n")


# ==================================================
# INTERACTIVE TERMINAL CHAT
# ==================================================

def run_terminal_chat(chunks, embeddings, model):

    print("\n" + "="*60)
    print("  MEDICAL CANCER QA CHATBOT")
    print("  Powered by LLaMA3 + Your 25 PDFs")
    print("="*60)
    print("  Type your question and press Enter")
    print("  Type 'exit' or 'quit' to stop")
    print("="*60 + "\n")

    while True:

        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye! Stay healthy!")
            break

        print("\nSearching your medical textbooks...")

        result = answer_question(
            question   = question,
            chunks     = chunks,
            embeddings = embeddings,
            model      = model
        )

        print_response(result)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # Step 1 — Load embedding model
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Model loaded ✅")

    # Step 2 — Load chunks + embeddings
    chunks, embeddings = load_embeddings_and_chunks()

    # Step 3 — Check Ollama
    print("Checking Ollama...")
    try:
        requests.get("http://localhost:11434", timeout=2)
        print(f"Ollama running ✅ (model: {MODEL_NAME})\n")
    except Exception:
        print("Ollama not running ⚠️")
        print("Run in another terminal: ollama serve")
        print("Using fallback mode\n")

    # Step 4 — Start interactive chat
    run_terminal_chat(chunks, embeddings, model)