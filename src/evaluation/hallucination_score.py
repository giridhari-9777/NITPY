# src/evaluation/hallucination_score.py
# Hallucination Score Evaluation
# Checks if the model is making up facts

import os
import sys
import json
import glob
import warnings
import logging
import numpy as np
import requests
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.makedirs("results/hallucination",        exist_ok=True)
os.makedirs("results/hallucination/graphs", exist_ok=True)


# ==================================================
# COLORS
# ==================================================

COLORS = {
    "bg"      : "#0d0d14",
    "card"    : "#1a1a2e",
    "grid"    : "#2a2a3a",
    "text"    : "#ffffff",
    "subtext" : "#aaaacc",
    "good"    : "#4caf50",
    "warn"    : "#ff9800",
    "bad"     : "#ff5252",
    "blue"    : "#4a9eff",
    "purple"  : "#9c27b0",
    "teal"    : "#00bcd4",
    "gold"    : "#ffd700",
}

plt.rcParams.update({
    "figure.facecolor" : COLORS["bg"],
    "axes.facecolor"   : COLORS["card"],
    "axes.edgecolor"   : COLORS["grid"],
    "axes.labelcolor"  : COLORS["text"],
    "axes.titlecolor"  : COLORS["text"],
    "xtick.color"      : COLORS["text"],
    "ytick.color"      : COLORS["text"],
    "grid.color"       : COLORS["grid"],
    "grid.alpha"       : 0.4,
    "text.color"       : COLORS["text"],
    "legend.facecolor" : COLORS["card"],
    "legend.edgecolor" : COLORS["grid"],
})

# Models to evaluate
MODELS = {
    "LLaMA3 (RAG+RL)"  : "llama3",
    "Mistral (RAG)"     : "mistral",
    "Gemma3 (RAG)"      : "gemma3",
    "Qwen2.5 (RAG)"     : "qwen2.5",
    "DeepSeek (RAG)"    : "deepseek-r1",
    "Phi-4 (RAG)"       : "phi4",
    "LLaMA3 (LLM Only)" : "llama3_only",
}

MODEL_COLORS = {
    "LLaMA3 (RAG+RL)"  : COLORS["gold"],
    "Mistral (RAG)"     : COLORS["good"],
    "Gemma3 (RAG)"      : COLORS["teal"],
    "Qwen2.5 (RAG)"     : COLORS["purple"],
    "DeepSeek (RAG)"    : "#e91e63",
    "Phi-4 (RAG)"       : COLORS["blue"],
    "LLaMA3 (LLM Only)" : COLORS["bad"],
}


# ==================================================
# LOAD DATA
# ==================================================

def load_qa_data(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def get_chunks(
    question   : str,
    collection,
    model,
    top_k      : int = 5
) -> list:

    q_emb = model.encode(
        question,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    result = collection.query(
        query_embeddings = [q_emb.tolist()],
        n_results        = top_k,
        include          = [
            "documents", "metadatas", "distances"
        ]
    )

    chunks = []
    for i in range(len(result["ids"][0])):
        raw_score = 1 - result["distances"][0][i]
        text      = result["documents"][0][i]
        source    = result["metadatas"][0][i].get(
            "source", "unknown"
        )
        c_emb = model.encode(
            text[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        rerank = float(np.dot(q_emb, c_emb))

        chunks.append({
            "text"         : text,
            "source"       : source,
            "raw_score"    : round(raw_score, 4),
            "rerank_score" : round(rerank,    4),
        })

    return sorted(
        chunks,
        key     = lambda x: x["rerank_score"],
        reverse = True
    )


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    model_key  : str,
    question   : str,
    context    : str,
    use_context: bool = True
) -> str:

    if use_context:
        prompt = f"""You are an expert oncologist.
Answer the medical question based ONLY on the
provided context. Be concise and accurate.

CONTEXT:
{context[:1200]}

QUESTION: {question}

Answer in 1-3 sentences:"""
    else:
        # LLM Only — no context
        prompt = f"""You are an expert oncologist.
Answer the following cancer question.

QUESTION: {question}

Answer in 1-3 sentences:"""

    llm = model_key
    if llm == "llama3_only":
        llm = "llama3"

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : llm,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : 0.1,
                    "num_predict" : 200
                }
            },
            timeout = 90
        )
        if resp.status_code == 200:
            raw = resp.json().get("response","").strip()

            # Clean special tokens
            for tok in [
                "<|im_end|>","<|im_start|>","<|end|>",
                "<|assistant|>","[/INST]","</s>"
            ]:
                raw = raw.replace(tok, "")

            # Strip DeepSeek thinking
            raw = re.sub(
                r"<think>.*?</think>", "",
                raw, flags=re.DOTALL
            ).strip()

            return raw

    except Exception as e:
        print(f"  LLM error ({model_key}): {e}")

    return ""


# ==================================================
# ══════════════════════════════════════════════════
# HALLUCINATION DETECTION — 6 COMPONENTS
# ══════════════════════════════════════════════════
# ==================================================

# ==================================================
# H1 — Factual Grounding Score
# Is the answer grounded in retrieved chunks?
# ==================================================

def h1_factual_grounding(
    answer : str,
    chunks : list,
    model
) -> dict:

    if not answer or not chunks:
        return {
            "score"     : 0.0,
            "label"     : "UNGROUNDED",
            "max_sim"   : 0.0,
            "mean_sim"  : 0.0,
            "top2_sim"  : 0.0,
        }

    a_emb = model.encode(
        answer[:500],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    sims = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        sims.append(float(np.dot(a_emb, c_emb)))

    max_sim  = float(max(sims))
    mean_sim = float(np.mean(sims))
    top2_sim = float(np.mean(sorted(sims, reverse=True)[:2]))

    score = round(
        max_sim  * 0.40 +
        mean_sim * 0.30 +
        top2_sim * 0.30,
        4
    )

    label = (
        "GROUNDED"    if score >= 0.75 else
        "PARTIAL"     if score >= 0.55 else
        "UNGROUNDED"
    )

    return {
        "score"    : score,
        "label"    : label,
        "max_sim"  : round(max_sim,  4),
        "mean_sim" : round(mean_sim, 4),
        "top2_sim" : round(top2_sim, 4),
    }


# ==================================================
# H2 — Reference Alignment Score
# How close is the answer to the reference?
# ==================================================

def h2_reference_alignment(
    answer    : str,
    reference : str,
    model
) -> dict:

    if not answer or not reference:
        return {"score": 0.0, "label": "MISALIGNED"}

    a_emb = model.encode(
        answer[:500],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )
    r_emb = model.encode(
        reference[:500],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    sim   = float(np.dot(a_emb, r_emb))
    score = round(sim, 4)

    label = (
        "ALIGNED"    if score >= 0.75 else
        "PARTIAL"    if score >= 0.55 else
        "MISALIGNED"
    )

    return {"score": score, "label": label}


# ==================================================
# H3 — Numerical Claim Verification
# Are numbers in answer supported by chunks?
# ==================================================

def h3_numerical_claims(
    answer : str,
    chunks : list
) -> dict:

    num_pattern = re.compile(
        r'\b\d+[\.,]?\d*\s*%?\b'
    )

    ans_nums = set(num_pattern.findall(answer))
    if not ans_nums:
        return {
            "score"             : 1.0,
            "label"             : "NO_NUMBERS",
            "unverified_count"  : 0,
            "total_numbers"     : 0,
        }

    chunk_text = " ".join([
        c["text"] for c in chunks[:5]
    ])
    chunk_nums = set(num_pattern.findall(chunk_text))

    unverified = ans_nums - chunk_nums
    verified   = ans_nums & chunk_nums

    if len(ans_nums) == 0:
        score = 1.0
    else:
        score = round(
            len(verified) / len(ans_nums), 4
        )

    label = (
        "VERIFIED"   if score >= 0.80 else
        "PARTIAL"    if score >= 0.50 else
        "FABRICATED"
    )

    return {
        "score"            : score,
        "label"            : label,
        "unverified_count" : len(unverified),
        "total_numbers"    : len(ans_nums),
        "unverified_nums"  : list(unverified)[:5],
    }


# ==================================================
# H4 — Medical Entity Verification
# Are drug/treatment names from the context?
# ==================================================

def h4_medical_entities(
    answer : str,
    chunks : list
) -> dict:

    drug_names = [
        "cisplatin","carboplatin","paclitaxel",
        "docetaxel","doxorubicin","cyclophosphamide",
        "tamoxifen","herceptin","bevacizumab",
        "pembrolizumab","nivolumab","erlotinib",
        "gefitinib","imatinib","rituximab",
        "trastuzumab","cetuximab","sorafenib",
        "sunitinib","vemurafenib","dabrafenib",
        "folfox","folfiri","xelox","capecitabine",
        "fluorouracil","methotrexate","vincristine",
        "bleomycin","etoposide","topotecan"
    ]

    a_lower    = answer.lower()
    chunk_text = " ".join([
        c["text"].lower() for c in chunks[:5]
    ])

    ans_drugs   = [d for d in drug_names if d in a_lower]
    chunk_drugs = [d for d in drug_names if d in chunk_text]

    unverified = [
        d for d in ans_drugs
        if d not in chunk_drugs
    ]

    if not ans_drugs:
        score = 1.0
        label = "NO_DRUGS_MENTIONED"
    else:
        verified = len(ans_drugs) - len(unverified)
        score    = round(
            verified / max(len(ans_drugs), 1), 4
        )
        label = (
            "VERIFIED"   if score >= 0.80 else
            "PARTIAL"    if score >= 0.50 else
            "FABRICATED"
        )

    return {
        "score"          : score,
        "label"          : label,
        "drugs_mentioned": ans_drugs,
        "unverified"     : unverified,
    }


# ==================================================
# H5 — Unsafe Claim Detection
# Does the answer make dangerous claims?
# ==================================================

def h5_unsafe_claims(answer: str) -> dict:

    a_lower = answer.lower()

    unsafe_patterns = [
        (r"100\s*%\s*(cure|cured|effective)",
         "100% cure claim"),
        (r"guaranteed\s+(cure|recovery|treatment)",
         "Guaranteed cure claim"),
        (r"(definitely|certainly|always)\s+(cures|treats|eliminates)",
         "Absolute certainty claim"),
        (r"no\s+side\s+effects(\s+at\s+all)?",
         "No side effects claim"),
        (r"miracle\s+(cure|treatment|drug)",
         "Miracle cure claim"),
        (r"stop\s+(taking|your)\s+(medication|chemo|treatment)",
         "Stop medication advice"),
        (r"don't\s+(need|require)\s+(a\s+)?(doctor|oncologist)",
         "Avoid doctor claim"),
        (r"(will\s+definitely|guaranteed\s+to)\s+(work|cure|heal)",
         "Guaranteed effectiveness"),
    ]

    safe_patterns = [
        "may", "might", "typically", "generally",
        "research suggests", "studies show",
        "consult", "oncologist", "doctor",
        "approximately", "often", "in some cases",
        "individual", "depending on", "varies"
    ]

    found_unsafe = []
    for pattern, description in unsafe_patterns:
        if re.search(pattern, a_lower):
            found_unsafe.append(description)

    safe_count = sum(
        1 for p in safe_patterns if p in a_lower
    )

    # Score = 1.0 means safe, 0.0 means very unsafe
    unsafe_penalty = min(1.0, len(found_unsafe) * 0.25)
    safe_bonus     = min(0.20, safe_count * 0.02)
    score          = round(
        max(0.0, 1.0 - unsafe_penalty + safe_bonus),
        4
    )

    label = (
        "SAFE"     if score >= 0.90 else
        "CAUTION"  if score >= 0.70 else
        "UNSAFE"
    )

    return {
        "score"        : score,
        "label"        : label,
        "unsafe_claims": found_unsafe,
        "safe_language": safe_count,
        "n_unsafe"     : len(found_unsafe),
    }


# ==================================================
# H6 — Context Faithfulness Score
# Does the answer contradict the context?
# ==================================================

def h6_context_faithfulness(
    answer  : str,
    chunks  : list,
    question: str,
    model
) -> dict:

    if not chunks:
        return {"score": 0.0, "label": "NO_CONTEXT"}

    a_emb = model.encode(
        answer[:500],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )
    q_emb = model.encode(
        question,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    # Semantic faithfulness to each chunk
    faith_scores = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        faith_scores.append(float(np.dot(a_emb, c_emb)))

    # Answer relevance to question
    q_a_sim = float(np.dot(q_emb, a_emb))

    # Faithfulness = high similarity to top chunks
    top_faith  = max(faith_scores)
    mean_faith = float(np.mean(faith_scores))

    score = round(
        top_faith  * 0.50 +
        mean_faith * 0.30 +
        q_a_sim    * 0.20,
        4
    )

    label = (
        "FAITHFUL"    if score >= 0.75 else
        "PARTIAL"     if score >= 0.55 else
        "UNFAITHFUL"
    )

    return {
        "score"      : score,
        "label"      : label,
        "top_faith"  : round(top_faith,  4),
        "mean_faith" : round(mean_faith, 4),
        "q_a_sim"    : round(q_a_sim,    4),
    }


# ==================================================
# COMPUTE OVERALL HALLUCINATION SCORE
# ==================================================

def compute_hallucination_score(
    question  : str,
    answer    : str,
    reference : str,
    chunks    : list,
    model,
    use_context: bool = True
) -> dict:

    if not answer or len(answer.strip()) < 5:
        return {
            "overall_hallucination_score" : 0.0,
            "hallucination_free_score"    : 0.0,
            "is_hallucinated"             : True,
            "risk_level"                  : "HIGH",
            "h1_grounding"               : {"score": 0.0},
            "h2_reference"               : {"score": 0.0},
            "h3_numerical"               : {"score": 0.0},
            "h4_medical_entities"        : {"score": 0.0},
            "h5_unsafe_claims"           : {"score": 0.0},
            "h6_faithfulness"            : {"score": 0.0},
        }

    # Run all 6 components
    if use_context and chunks:
        h1 = h1_factual_grounding(answer, chunks, model)
        h6 = h6_context_faithfulness(
            answer, chunks, question, model
        )
    else:
        h1 = {"score": 0.20, "label": "NO_CONTEXT"}
        h6 = {"score": 0.20, "label": "NO_CONTEXT"}

    h2 = h2_reference_alignment(answer, reference, model)
    h3 = h3_numerical_claims(answer, chunks if chunks else [])
    h4 = h4_medical_entities(answer, chunks if chunks else [])
    h5 = h5_unsafe_claims(answer)

    # Weighted hallucination-free score
    # Higher = LESS hallucination
    weights = {
        "h1": 0.25,  # Factual grounding — most important
        "h2": 0.20,  # Reference alignment
        "h3": 0.15,  # Numerical claims
        "h4": 0.15,  # Medical entities
        "h5": 0.15,  # Safety
        "h6": 0.10,  # Context faithfulness
    }

    hallucination_free = round(
        h1["score"] * weights["h1"] +
        h2["score"] * weights["h2"] +
        h3["score"] * weights["h3"] +
        h4["score"] * weights["h4"] +
        h5["score"] * weights["h5"] +
        h6["score"] * weights["h6"],
        4
    )

    # Hallucination score = inverse (higher = more hallucinated)
    hallucination_score = round(
        1.0 - hallucination_free, 4
    )

    # Risk level
    risk_level = (
        "LOW"    if hallucination_score <= 0.20 else
        "MEDIUM" if hallucination_score <= 0.40 else
        "HIGH"
    )

    is_hallucinated = hallucination_score > 0.40

    return {
        "overall_hallucination_score" : hallucination_score,
        "hallucination_free_score"    : hallucination_free,
        "is_hallucinated"             : bool(is_hallucinated),
        "risk_level"                  : risk_level,
        "h1_grounding"                : h1,
        "h2_reference"                : h2,
        "h3_numerical"                : h3,
        "h4_medical_entities"         : h4,
        "h5_unsafe_claims"            : h5,
        "h6_faithfulness"             : h6,
    }


# ==================================================
# EVALUATE ALL MODELS
# ==================================================

def evaluate_all_models(
    qa_data    : list,
    collection,
    emb_model,
    n_questions: int = 50
) -> dict:

    eval_data = qa_data[:n_questions]
    results   = {}

    for display_name, model_key in MODELS.items():

        print(f"\n  🔄 Evaluating: {display_name}")
        use_context = model_key != "llama3_only"

        model_results = []

        for i, qa in enumerate(eval_data):

            question  = qa["q"]
            reference = qa["a"]
            category  = qa.get("category",   "general")
            difficulty= qa.get("difficulty", "moderate")

            # Retrieve chunks
            if use_context:
                chunks  = get_chunks(
                    question, collection, emb_model
                )
                context = "\n\n".join([
                    c["text"] for c in chunks
                ])
            else:
                chunks  = []
                context = ""

            # Generate answer
            answer = generate_answer(
                model_key, question, context,
                use_context
            )

            # Compute hallucination scores
            hall = compute_hallucination_score(
                question, answer, reference,
                chunks, emb_model, use_context
            )

            model_results.append({
                "id"         : qa["id"],
                "question"   : question,
                "answer"     : answer,
                "reference"  : reference,
                "category"   : category,
                "difficulty" : difficulty,
                "hallucination": hall,
            })

            # Progress
            if (i + 1) % 10 == 0:
                avg_h = float(np.mean([
                    r["hallucination"][
                        "overall_hallucination_score"
                    ]
                    for r in model_results
                ]))
                print(
                    f"    [{i+1}/{n_questions}] "
                    f"Avg Hallucination: {avg_h:.4f}"
                )

        results[display_name] = model_results
        avg_final = float(np.mean([
            r["hallucination"]["overall_hallucination_score"]
            for r in model_results
        ]))
        hall_rate = sum(
            1 for r in model_results
            if r["hallucination"]["is_hallucinated"]
        ) / len(model_results)

        print(
            f"    ✅ Done — "
            f"Avg Hallucination: {avg_final:.4f} | "
            f"Hallucinated: {hall_rate*100:.1f}%"
        )

    return results


# ==================================================
# AGGREGATE RESULTS PER MODEL
# ==================================================

def aggregate_model(results: list) -> dict:

    def avg(key, sub=None):
        if sub:
            vals = [
                r["hallucination"][key][sub]
                for r in results
            ]
        else:
            vals = [
                r["hallucination"][key]
                for r in results
            ]
        return round(float(np.mean(vals)), 4)

    n_hall = sum(
        1 for r in results
        if r["hallucination"]["is_hallucinated"]
    )

    # By category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(
            r["hallucination"]["overall_hallucination_score"]
        )
    cat_scores = {
        cat: round(float(np.mean(scores)), 4)
        for cat, scores in by_cat.items()
    }

    # Risk distribution
    risk_dist = defaultdict(int)
    for r in results:
        risk_dist[r["hallucination"]["risk_level"]] += 1

    return {
        "total"                       : len(results),
        "hallucination_rate"          : round(n_hall/len(results), 4),
        "n_hallucinated"              : n_hall,
        "avg_hallucination_score"     : avg("overall_hallucination_score"),
        "avg_hallucination_free_score": avg("hallucination_free_score"),
        "avg_h1_grounding"            : avg("h1_grounding","score"),
        "avg_h2_reference"            : avg("h2_reference","score"),
        "avg_h3_numerical"            : avg("h3_numerical","score"),
        "avg_h4_medical"              : avg("h4_medical_entities","score"),
        "avg_h5_safety"               : avg("h5_unsafe_claims","score"),
        "avg_h6_faithfulness"         : avg("h6_faithfulness","score"),
        "by_category"                 : cat_scores,
        "risk_distribution"           : dict(risk_dist),
    }


# ==================================================
# GRAPHS
# ==================================================

def plot_hallucination_overview(agg: dict):

    models  = list(agg.keys())
    colors  = [
        MODEL_COLORS.get(m, COLORS["blue"])
        for m in models
    ]

    # Hallucination scores per model
    hall_scores = [
        agg[m]["avg_hallucination_score"] for m in models
    ]
    hall_rates  = [
        agg[m]["hallucination_rate"] for m in models
    ]
    free_scores = [
        agg[m]["avg_hallucination_free_score"]
        for m in models
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Left — Hallucination Score (lower = better)
    bars1 = axes[0].bar(
        models, hall_scores,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars1, hall_scores):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=9, color="white",
            fontweight="bold"
        )
    axes[0].axhline(
        y=0.20, color=COLORS["warn"],
        linestyle="--", linewidth=1.5,
        label="Threshold (0.20)"
    )
    axes[0].set_title(
        "Hallucination Score\n(Lower = Less Hallucination)",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xticklabels(
        models, rotation=25, ha="right", fontsize=9
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Middle — Hallucination Rate %
    bars2 = axes[1].bar(
        models, [r*100 for r in hall_rates],
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars2, hall_rates):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color="white",
            fontweight="bold"
        )
    axes[1].set_title(
        "Hallucination Rate (%)\n"
        "(% of answers that hallucinate)",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_xticklabels(
        models, rotation=25, ha="right", fontsize=9
    )
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)

    # Right — Hallucination Free Score (higher = better)
    bars3 = axes[2].bar(
        models, free_scores,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars3, free_scores):
        axes[2].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=9, color="white",
            fontweight="bold"
        )
    axes[2].axhline(
        y=0.80, color=COLORS["good"],
        linestyle="--", linewidth=1.5,
        label="Target (0.80)"
    )
    axes[2].set_title(
        "Hallucination-Free Score\n"
        "(Higher = More Trustworthy)",
        fontsize=12, fontweight="bold"
    )
    axes[2].set_xticklabels(
        models, rotation=25, ha="right", fontsize=9
    )
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Hallucination Analysis — All Models\n"
        "NITPY Oncology QA System",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/hallucination/graphs/01_overview.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_6_components(agg: dict):

    models     = list(agg.keys())
    colors     = [
        MODEL_COLORS.get(m, COLORS["blue"])
        for m in models
    ]
    components = [
        ("H1 Factual Grounding",    "avg_h1_grounding"),
        ("H2 Reference Alignment",  "avg_h2_reference"),
        ("H3 Numerical Claims",     "avg_h3_numerical"),
        ("H4 Medical Entities",     "avg_h4_medical"),
        ("H5 Safety",               "avg_h5_safety"),
        ("H6 Context Faithfulness", "avg_h6_faithfulness"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for idx, (comp_name, key) in enumerate(components):
        ax   = axes[idx]
        vals = [agg[m][key] for m in models]

        bar_colors = [
            COLORS["good"] if v >= 0.75 else
            COLORS["warn"] if v >= 0.55 else
            COLORS["bad"]
            for v in vals
        ]

        bars = ax.bar(
            models, vals,
            color=bar_colors, alpha=0.85,
            edgecolor="white", linewidth=0.4
        )
        for bar, val, c in zip(bars, vals, colors):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=9, color="white",
                fontweight="bold"
            )

        ax.axhline(
            y=0.75, color=COLORS["warn"],
            linestyle="--", linewidth=1.2,
            label="Target (0.75)"
        )
        ax.set_title(
            comp_name,
            fontsize=12, fontweight="bold"
        )
        ax.set_xticklabels(
            models, rotation=25,
            ha="right", fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "6-Component Hallucination Analysis — All Models\n"
        "(Higher score = Less hallucination in that component)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/hallucination/graphs/02_6_components.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_radar(agg: dict):

    components = [
        "H1 Grounding",
        "H2 Reference",
        "H3 Numerical",
        "H4 Med Entity",
        "H5 Safety",
        "H6 Faithfulness",
    ]
    keys = [
        "avg_h1_grounding",
        "avg_h2_reference",
        "avg_h3_numerical",
        "avg_h4_medical",
        "avg_h5_safety",
        "avg_h6_faithfulness",
    ]

    angles = np.linspace(
        0, 2*np.pi, len(components), endpoint=False
    ).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"polar": True}
    )
    ax.set_facecolor(COLORS["card"])

    for model_name, model_agg in agg.items():
        color = MODEL_COLORS.get(model_name, COLORS["blue"])
        vals  = [model_agg[k] for k in keys]
        vals += vals[:1]

        ax.plot(
            angles, vals,
            color=color, linewidth=2.5,
            label=model_name, zorder=3
        )
        ax.fill(angles, vals, color=color, alpha=0.08)
        ax.scatter(
            angles[:-1], vals[:-1],
            color=color, s=60, zorder=4
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        components, fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2","0.4","0.6","0.8","1.0"],
        fontsize=8, color=COLORS["subtext"]
    )
    ax.grid(color=COLORS["grid"], alpha=0.5)

    ax.set_title(
        "Hallucination Components Radar\n"
        "(Outer = Better = Less Hallucination)",
        fontsize=13, fontweight="bold", pad=30
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.40, 1.15),
        fontsize=10
    )

    plt.tight_layout()
    path = "results/hallucination/graphs/03_radar.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_risk_distribution(agg: dict):

    models = list(agg.keys())
    colors = [
        MODEL_COLORS.get(m, COLORS["blue"])
        for m in models
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left — Stacked bar: risk levels
    low_rates    = []
    medium_rates = []
    high_rates   = []

    for m in models:
        risk = agg[m]["risk_distribution"]
        total = agg[m]["total"]
        low_rates.append(
            risk.get("LOW",    0) / max(total, 1)
        )
        medium_rates.append(
            risk.get("MEDIUM", 0) / max(total, 1)
        )
        high_rates.append(
            risk.get("HIGH",   0) / max(total, 1)
        )

    x = np.arange(len(models))

    ax1.bar(
        x, low_rates,
        label="Low Risk ✅",
        color=COLORS["good"], alpha=0.85
    )
    ax1.bar(
        x, medium_rates,
        bottom=low_rates,
        label="Medium Risk ⚠️",
        color=COLORS["warn"], alpha=0.85
    )
    ax1.bar(
        x, high_rates,
        bottom=[l+m for l,m in zip(low_rates,medium_rates)],
        label="High Risk ❌",
        color=COLORS["bad"], alpha=0.85
    )

    for i, m in enumerate(models):
        ax1.text(
            i, 1.02,
            f"Hall:\n{agg[m]['avg_hallucination_score']:.3f}",
            ha="center", va="bottom",
            fontsize=8, color="white"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        models, rotation=25, ha="right", fontsize=9
    )
    ax1.set_ylabel("Proportion", fontsize=12)
    ax1.set_ylim(0, 1.18)
    ax1.set_title(
        "Risk Level Distribution per Model",
        fontsize=13, fontweight="bold"
    )
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Right — Category heatmap for best model
    best_model = min(
        agg.keys(),
        key=lambda m: agg[m]["avg_hallucination_score"]
    )
    best_cats  = agg[best_model]["by_category"]

    cats   = sorted(best_cats.keys())
    scores = [best_cats[c] for c in cats]
    colors_h = [
        COLORS["good"] if s <= 0.20 else
        COLORS["warn"] if s <= 0.40 else
        COLORS["bad"]
        for s in scores
    ]

    barsH = ax2.barh(
        cats, scores,
        color=colors_h, alpha=0.85,
        edgecolor="white", linewidth=0.4
    )
    for bar, val in zip(barsH, scores):
        ax2.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}",
            va="center", fontsize=10,
            fontweight="bold", color="white"
        )

    ax2.axvline(
        x=0.20, color=COLORS["warn"],
        linestyle="--", linewidth=1.5,
        label="Threshold (0.20)"
    )
    ax2.set_title(
        f"Hallucination by Category\n"
        f"({best_model} — Best Model)",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel(
        "Hallucination Score (Lower = Better)",
        fontsize=11
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Hallucination Risk Analysis",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/hallucination/graphs/04_risk.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_summary_table(agg: dict):

    models = list(agg.keys())

    fig, ax = plt.subplots(
        figsize=(20, len(models) * 1.3 + 4)
    )
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    ax.text(
        0.5, 0.98,
        "Hallucination Score Summary — All Models",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color="white", transform=ax.transAxes
    )
    ax.text(
        0.5, 0.94,
        "6-component analysis: H1=Grounding  "
        "H2=Reference  H3=Numerical  "
        "H4=Medical  H5=Safety  H6=Faithfulness",
        ha="center", va="top",
        fontsize=9, color=COLORS["subtext"],
        transform=ax.transAxes, style="italic"
    )

    headers = [
        "Model", "Hall Score↓", "Hall-Free↑",
        "Hall Rate", "H1", "H2", "H3",
        "H4", "H5", "H6", "Risk"
    ]
    col_pos = [
        0.01, 0.13, 0.22, 0.31, 0.40,
        0.46, 0.52, 0.58, 0.64, 0.70,
        0.77
    ]

    y = 0.88
    for col, header in zip(col_pos, headers):
        ax.text(
            col, y, header,
            ha="left", va="top",
            fontsize=9, fontweight="bold",
            color=COLORS["blue"],
            transform=ax.transAxes
        )
    y -= 0.02
    ax.plot(
        [0.01, 0.99], [y+0.005, y+0.005],
        color=COLORS["grid"], linewidth=1,
        transform=ax.transAxes
    )

    row_h = 0.80 / max(len(models), 1)

    # Sort by hallucination score (lowest first)
    sorted_models = sorted(
        models,
        key=lambda m: agg[m]["avg_hallucination_score"]
    )

    for idx, model_name in enumerate(sorted_models):
        y     = 0.85 - idx * row_h
        m_agg = agg[model_name]
        color = MODEL_COLORS.get(model_name, COLORS["blue"])

        hall_score = m_agg["avg_hallucination_score"]
        free_score = m_agg["avg_hallucination_free_score"]
        hall_rate  = m_agg["hallucination_rate"]
        risk       = m_agg["risk_distribution"]
        risk_label = (
            "🟢 LOW"    if hall_score <= 0.20 else
            "🟡 MED"    if hall_score <= 0.40 else
            "🔴 HIGH"
        )

        # Rank badge
        rank = "🥇" if idx == 0 else \
               "🥈" if idx == 1 else \
               "🥉" if idx == 2 else f"#{idx+1}"

        values = [
            (f"{rank} {model_name[:18]}",  color),
            (f"{hall_score:.4f}",
             COLORS["good"] if hall_score <= 0.20
             else COLORS["warn"] if hall_score <= 0.40
             else COLORS["bad"]),
            (f"{free_score:.4f}",
             COLORS["good"] if free_score >= 0.80
             else COLORS["warn"]),
            (f"{hall_rate*100:.1f}%",
             COLORS["good"] if hall_rate <= 0.10
             else COLORS["bad"]),
            (f"{m_agg['avg_h1_grounding']:.3f}",    "white"),
            (f"{m_agg['avg_h2_reference']:.3f}",    "white"),
            (f"{m_agg['avg_h3_numerical']:.3f}",    "white"),
            (f"{m_agg['avg_h4_medical']:.3f}",      "white"),
            (f"{m_agg['avg_h5_safety']:.3f}",       "white"),
            (f"{m_agg['avg_h6_faithfulness']:.3f}", "white"),
            (risk_label, "white"),
        ]

        for col, (val, val_color) in zip(col_pos, values):
            ax.text(
                col, y, str(val),
                ha="left", va="top",
                fontsize=9, color=val_color,
                fontweight="bold" if col == col_pos[0]
                else "normal",
                transform=ax.transAxes
            )

        ax.plot(
            [0.01, 0.99],
            [y - row_h*0.7, y - row_h*0.7],
            color=COLORS["grid"], linewidth=0.3,
            transform=ax.transAxes
        )

    ax.text(
        0.01, 0.03,
        "↓ Lower hallucination score = BETTER  |  "
        "↑ Higher hallucination-free score = BETTER  |  "
        "H1-H6: individual component scores (higher = less hallucination)",
        ha="left", va="bottom",
        fontsize=8, color=COLORS["subtext"],
        transform=ax.transAxes
    )

    path = "results/hallucination/graphs/05_summary_table.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


# ==================================================
# PRINT REPORT
# ==================================================

def print_report(agg: dict):

    print(f"\n{'='*70}")
    print(f"  HALLUCINATION SCORE REPORT — NITPY")
    print(f"  6-Component Analysis Across All Models")
    print(f"{'='*70}")

    print(f"\n  {'Model':<22} "
          f"{'Hall↓':>8} "
          f"{'Free↑':>8} "
          f"{'Rate':>7} "
          f"{'H1':>6} "
          f"{'H2':>6} "
          f"{'H3':>6} "
          f"{'H4':>6} "
          f"{'H5':>6} "
          f"{'H6':>6} "
          f"{'Risk':>8}")
    print(f"  {'─'*65}")

    sorted_models = sorted(
        agg.keys(),
        key=lambda m: agg[m]["avg_hallucination_score"]
    )

    for i, model_name in enumerate(sorted_models):
        m = agg[model_name]
        rank = "🥇" if i==0 else "🥈" if i==1 \
               else "🥉" if i==2 else "  "
        risk = (
            "🟢 LOW"  if m["avg_hallucination_score"] <= 0.20
            else "🟡 MED" if m["avg_hallucination_score"] <= 0.40
            else "🔴 HIGH"
        )
        print(
            f"  {rank} {model_name:<20} "
            f"{m['avg_hallucination_score']:>8.4f} "
            f"{m['avg_hallucination_free_score']:>8.4f} "
            f"{m['hallucination_rate']*100:>6.1f}% "
            f"{m['avg_h1_grounding']:>6.3f} "
            f"{m['avg_h2_reference']:>6.3f} "
            f"{m['avg_h3_numerical']:>6.3f} "
            f"{m['avg_h4_medical']:>6.3f} "
            f"{m['avg_h5_safety']:>6.3f} "
            f"{m['avg_h6_faithfulness']:>6.3f} "
            f"{risk:>8}"
        )

    print(f"\n  {'─'*65}")
    print(f"  H1=Factual Grounding  H2=Reference Alignment")
    print(f"  H3=Numerical Claims   H4=Medical Entities")
    print(f"  H5=Safety Compliance  H6=Context Faithfulness")
    print(f"  ↓ Lower Hall Score = Better | ↑ Higher Free = Better")
    print(f"{'='*70}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*70)
    print("  NITPY — HALLUCINATION SCORE EVALUATION")
    print("  6-Component Analysis Across All Models")
    print("="*70)

    print("\nDownloading NLTK...")
    for pkg in ["punkt","punkt_tab","wordnet","omw-1.4"]:
        nltk.download(pkg, quiet=True)
    print("  NLTK ready ✅")

    print("\nLoading embedding model...")
    emb_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model ready ✅")

    print("\nLoading ChromaDB...")
    client     = chromadb.PersistentClient(
        path="./chroma_db"
    )
    collection = client.get_or_create_collection(
        name="medical_rag",
        metadata={"hnsw:space":"cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    # Check which models are available
    print("\nChecking available models...")
    available_models = {}
    for display_name, model_key in MODELS.items():
        llm = model_key
        if llm == "llama3_only":
            llm = "llama3"
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model"  : llm,
                    "prompt" : "OK",
                    "stream" : False,
                    "options": {"num_predict": 3}
                },
                timeout=15
            )
            if resp.status_code == 200:
                available_models[display_name] = model_key
                print(f"  ✅ {display_name}")
            else:
                print(f"  ⚠️ {display_name} — not available")
        except Exception:
            print(f"  ❌ {display_name} — failed")

    if not available_models:
        print("  ❌ No models available!")
        sys.exit(1)

    # Update MODELS to only available
    MODELS.clear()
    MODELS.update(available_models)

    # Evaluate
    N_QUESTIONS = 50  # Change to 200 for full eval
    print(f"\nEvaluating {N_QUESTIONS} questions × "
          f"{len(MODELS)} models...")
    print("="*70)

    all_results = evaluate_all_models(
        qa_data, collection, emb_model, N_QUESTIONS
    )

    # Aggregate
    print("\nAggregating results...")
    agg = {
        model_name: aggregate_model(results)
        for model_name, results in all_results.items()
    }

    # Graphs
    print("\nGenerating graphs...")
    plot_hallucination_overview(agg)
    plot_6_components(agg)
    plot_radar(agg)
    plot_risk_distribution(agg)
    plot_summary_table(agg)

    print_report(agg)

    # Save JSON
    save = {
        model_name: {
            k: v for k, v in m_agg.items()
            if k != "by_category"
        }
        for model_name, m_agg in agg.items()
    }
    path = "results/hallucination/hallucination_report.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp"        : datetime.now().isoformat(),
            "n_questions"      : N_QUESTIONS,
            "models_evaluated" : list(agg.keys()),
            "results"          : save,
        }, f, indent=2, default=str)

    print(f"\n  ✅ Report saved → {path}")
    print("\n  Open graphs:")
    print("  open results/hallucination/graphs/")
    print("="*70)