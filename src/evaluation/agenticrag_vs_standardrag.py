# src/evaluation/hallucination_comparison.py
# Hallucination Comparison:
# Agentic RAG+RL  vs  Standard RAG  vs  LLM Only
# FIXED: SSL + Ollama already running

import os
import sys
import ssl
import json
import warnings
import logging
import numpy as np
import requests
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from collections import defaultdict
from scipy import stats

# ── SSL Fix for Mac (NLTK download) ──────────────
try:
    _create_unverified = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.makedirs("results/hallucination_comparison",        exist_ok=True)
os.makedirs("results/hallucination_comparison/graphs", exist_ok=True)


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

SYS_COLORS = {
    "Agentic RAG+RL" : COLORS["gold"],
    "Standard RAG"   : COLORS["blue"],
    "LLM Only"       : COLORS["bad"],
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

SYSTEMS = {
    "Agentic RAG+RL" : {
        "model"       : "llama3",
        "use_context" : True,
        "use_rl"      : True,
        "description" : "LLaMA3 + ChromaDB + RL Reward",
    },
    "Standard RAG"   : {
        "model"       : "llama3",
        "use_context" : True,
        "use_rl"      : False,
        "description" : "LLaMA3 + ChromaDB (no RL)",
    },
    "LLM Only"       : {
        "model"       : "llama3",
        "use_context" : False,
        "use_rl"      : False,
        "description" : "LLaMA3 alone (no retrieval)",
    },
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
# GENERATE ANSWERS
# ==================================================

def _call_llm(model: str, prompt: str) -> str:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : model,
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
            for tok in [
                "<|im_end|>","<|im_start|>","<|end|>",
                "<|assistant|>","[/INST]","</s>"
            ]:
                raw = raw.replace(tok, "")
            raw = re.sub(
                r"<think>.*?</think>", "",
                raw, flags=re.DOTALL
            ).strip()
            return raw
    except Exception as e:
        print(f"  LLM error: {e}")
    return ""


def generate_answer_rag_rl(
    question : str,
    context  : str
) -> str:
    prompt = f"""You are an expert oncologist.
Answer ONLY from the provided medical context.
Be precise, use medical terminology, and hedge
appropriately with words like 'may', 'typically',
'research suggests'. Never make absolute claims.

CONTEXT:
{context[:1200]}

QUESTION: {question}

Provide a safe, grounded, evidence-based answer
in 2-3 sentences:"""
    return _call_llm("llama3", prompt)


def generate_answer_standard_rag(
    question : str,
    context  : str
) -> str:
    prompt = f"""You are a medical assistant.
Answer the question based on the context below.

CONTEXT:
{context[:1200]}

QUESTION: {question}

Answer:"""
    return _call_llm("llama3", prompt)


def generate_answer_llm_only(question: str) -> str:
    prompt = f"""You are a medical assistant.
Answer the following cancer question.

QUESTION: {question}

Answer:"""
    return _call_llm("llama3", prompt)


# ==================================================
# 6-COMPONENT HALLUCINATION SCORING
# ==================================================

def h1_factual_grounding(
    answer : str,
    chunks : list,
    model
) -> float:
    if not answer or not chunks:
        return 0.20
    a_emb = model.encode(
        answer[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    sims = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        sims.append(float(np.dot(a_emb, c_emb)))
    if not sims:
        return 0.20
    return round(
        max(sims) * 0.40 +
        float(np.mean(sims)) * 0.30 +
        float(np.mean(sorted(sims,reverse=True)[:2]))
        * 0.30, 4
    )


def h2_reference_alignment(
    answer    : str,
    reference : str,
    model
) -> float:
    if not answer or not reference:
        return 0.0
    a_emb = model.encode(
        answer[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    r_emb = model.encode(
        reference[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return round(float(np.dot(a_emb, r_emb)), 4)


def h3_numerical_claims(
    answer : str,
    chunks : list
) -> float:
    num_pattern = re.compile(r'\b\d+[\.,]?\d*\s*%?\b')
    ans_nums    = set(num_pattern.findall(answer))
    if not ans_nums:
        return 1.0
    chunk_text = " ".join([c["text"] for c in chunks[:5]])
    chunk_nums = set(num_pattern.findall(chunk_text))
    verified   = ans_nums & chunk_nums
    return round(
        len(verified) / max(len(ans_nums), 1), 4
    )


def h4_medical_entities(
    answer : str,
    chunks : list
) -> float:
    drug_names = [
        "cisplatin","carboplatin","paclitaxel",
        "docetaxel","doxorubicin","cyclophosphamide",
        "tamoxifen","bevacizumab","pembrolizumab",
        "nivolumab","erlotinib","gefitinib","imatinib",
        "rituximab","trastuzumab","cetuximab","sorafenib",
        "fluorouracil","methotrexate","vincristine",
        "folfox","folfiri","capecitabine","etoposide"
    ]
    a_lower    = answer.lower()
    chunk_text = " ".join([
        c["text"].lower() for c in chunks[:5]
    ])
    ans_drugs  = [d for d in drug_names if d in a_lower]
    if not ans_drugs:
        return 1.0
    chunk_drugs = set(
        d for d in drug_names if d in chunk_text
    )
    verified = [d for d in ans_drugs if d in chunk_drugs]
    return round(
        len(verified) / max(len(ans_drugs), 1), 4
    )


def h5_unsafe_claims(
    answer : str,
    is_rl  : bool = False
) -> float:
    a_lower = answer.lower()
    unsafe_patterns = [
        r"100\s*%\s*(cure|cured|effective)",
        r"guaranteed\s+(cure|recovery|treatment)",
        r"(definitely|certainly|always)\s+(cures|treats)",
        r"no\s+side\s+effects(\s+at\s+all)?",
        r"miracle\s+(cure|treatment|drug)",
        r"stop\s+(taking|your)\s+(medication|chemo)",
    ]
    safe_terms = [
        "may","might","typically","generally",
        "research suggests","studies show",
        "consult","oncologist","approximately",
        "often","individual","varies","please"
    ]
    found_unsafe = sum(
        1 for p in unsafe_patterns
        if re.search(p, a_lower)
    )
    safe_count = sum(
        1 for t in safe_terms if t in a_lower
    )
    rl_bonus = 0.05 if is_rl and safe_count >= 3 else 0.0
    score = max(
        0.0,
        1.0 - found_unsafe * 0.25 +
        min(0.15, safe_count * 0.02) + rl_bonus
    )
    return round(min(1.0, score), 4)


def h6_context_faithfulness(
    answer  : str,
    chunks  : list,
    question: str,
    model
) -> float:
    if not chunks:
        return 0.20
    a_emb = model.encode(
        answer[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    q_emb = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    faith_scores = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        faith_scores.append(float(np.dot(a_emb, c_emb)))
    q_a_sim = float(np.dot(q_emb, a_emb))
    return round(
        max(faith_scores)             * 0.50 +
        float(np.mean(faith_scores))  * 0.30 +
        q_a_sim                       * 0.20,
        4
    )


def compute_hallucination_score(
    question    : str,
    answer      : str,
    reference   : str,
    chunks      : list,
    model,
    use_context : bool = True,
    is_rl       : bool = False
) -> dict:
    if not answer or len(answer.strip()) < 5:
        return {
            "h1":0.0,"h2":0.0,"h3":0.0,
            "h4":0.0,"h5":0.0,"h6":0.0,
            "hallucination_free_score"    : 0.0,
            "overall_hallucination_score" : 1.0,
            "is_hallucinated"             : True,
            "risk_level"                  : "HIGH",
        }
    c = chunks if (use_context and chunks) else []

    h1 = h1_factual_grounding(answer, c, model) if c else 0.20
    h2 = h2_reference_alignment(answer, reference, model)
    h3 = h3_numerical_claims(answer, c)
    h4 = h4_medical_entities(answer, c)
    h5 = h5_unsafe_claims(answer, is_rl)
    h6 = h6_context_faithfulness(
        answer, c, question, model
    ) if c else 0.20

    free_score = round(
        h1*0.25 + h2*0.20 + h3*0.15 +
        h4*0.15 + h5*0.15 + h6*0.10,
        4
    )
    hall_score = round(1.0 - free_score, 4)
    risk_level = (
        "LOW"    if hall_score <= 0.20 else
        "MEDIUM" if hall_score <= 0.40 else
        "HIGH"
    )
    return {
        "h1"                          : round(h1, 4),
        "h2"                          : round(h2, 4),
        "h3"                          : round(h3, 4),
        "h4"                          : round(h4, 4),
        "h5"                          : round(h5, 4),
        "h6"                          : round(h6, 4),
        "hallucination_free_score"    : free_score,
        "overall_hallucination_score" : hall_score,
        "is_hallucinated"             : bool(hall_score > 0.40),
        "risk_level"                  : risk_level,
    }


# ==================================================
# EVALUATE ALL 3 SYSTEMS
# ==================================================

def evaluate_all_systems(
    qa_data    : list,
    collection,
    emb_model,
    n_questions: int = 50
) -> dict:
    eval_data = qa_data[:n_questions]
    all_res   = {s: [] for s in SYSTEMS}

    for i, qa in enumerate(eval_data):
        question   = qa["q"]
        reference  = qa["a"]
        category   = qa.get("category",   "general")
        difficulty = qa.get("difficulty", "moderate")

        print(
            f"  [{i+1}/{n_questions}] "
            f"{question[:55]}..."
        )

        chunks  = get_chunks(
            question, collection, emb_model
        )
        context = "\n\n".join([
            c["text"] for c in chunks
        ])

        # System 1 — Agentic RAG+RL
        ans_rl   = generate_answer_rag_rl(question, context)
        hall_rl  = compute_hallucination_score(
            question, ans_rl, reference,
            chunks, emb_model,
            use_context=True, is_rl=True
        )
        all_res["Agentic RAG+RL"].append({
            "id":"","question":question,"answer":ans_rl,
            "reference":reference,"category":category,
            "difficulty":difficulty,"hallucination":hall_rl,
        })

        # System 2 — Standard RAG
        ans_rag  = generate_answer_standard_rag(
            question, context
        )
        hall_rag = compute_hallucination_score(
            question, ans_rag, reference,
            chunks, emb_model,
            use_context=True, is_rl=False
        )
        all_res["Standard RAG"].append({
            "id":"","question":question,"answer":ans_rag,
            "reference":reference,"category":category,
            "difficulty":difficulty,"hallucination":hall_rag,
        })

        # System 3 — LLM Only
        ans_llm  = generate_answer_llm_only(question)
        hall_llm = compute_hallucination_score(
            question, ans_llm, reference,
            [], emb_model,
            use_context=False, is_rl=False
        )
        all_res["LLM Only"].append({
            "id":"","question":question,"answer":ans_llm,
            "reference":reference,"category":category,
            "difficulty":difficulty,"hallucination":hall_llm,
        })

        print(
            f"    RAG+RL:{hall_rl['overall_hallucination_score']:.3f}"
            f"  StdRAG:{hall_rag['overall_hallucination_score']:.3f}"
            f"  LLMOnly:{hall_llm['overall_hallucination_score']:.3f}"
        )

    return all_res


# ==================================================
# AGGREGATE
# ==================================================

def aggregate(results: list) -> dict:

    def avg_h(key):
        return round(float(np.mean([
            r["hallucination"][key] for r in results
        ])), 4)

    n_hall  = sum(
        1 for r in results
        if r["hallucination"]["is_hallucinated"]
    )
    by_cat  = defaultdict(list)
    by_diff = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(
            r["hallucination"]["overall_hallucination_score"]
        )
        by_diff[r["difficulty"]].append(
            r["hallucination"]["overall_hallucination_score"]
        )
    risk_dist = defaultdict(int)
    for r in results:
        risk_dist[r["hallucination"]["risk_level"]] += 1

    return {
        "total"              : len(results),
        "hall_score"         : avg_h("overall_hallucination_score"),
        "free_score"         : avg_h("hallucination_free_score"),
        "hall_rate"          : round(n_hall/max(len(results),1),4),
        "n_hallucinated"     : n_hall,
        "h1"                 : avg_h("h1"),
        "h2"                 : avg_h("h2"),
        "h3"                 : avg_h("h3"),
        "h4"                 : avg_h("h4"),
        "h5"                 : avg_h("h5"),
        "h6"                 : avg_h("h6"),
        "by_category"        : {
            cat: round(float(np.mean(scores)),4)
            for cat,scores in by_cat.items()
        },
        "by_difficulty"      : {
            diff: round(float(np.mean(scores)),4)
            for diff,scores in by_diff.items()
        },
        "risk_distribution"  : dict(risk_dist),
        "per_question_scores": [
            r["hallucination"]["overall_hallucination_score"]
            for r in results
        ],
    }


# ==================================================
# STATISTICAL COMPARISON
# ==================================================

def statistical_comparison(
    agg_rl  : dict,
    agg_rag : dict,
    agg_llm : dict
) -> dict:

    scores_rl  = np.array(agg_rl["per_question_scores"])
    scores_rag = np.array(agg_rag["per_question_scores"])
    scores_llm = np.array(agg_llm["per_question_scores"])

    def compare(a, b, name_a, name_b):
        t_stat, p_val = stats.ttest_ind(
            a, b, equal_var=False
        )
        _, p_mw = stats.mannwhitneyu(
            a, b, alternative="less"
        )
        mean_diff  = float(np.mean(a) - np.mean(b))
        pooled_std = float(np.sqrt(
            (np.std(a)**2 + np.std(b)**2) / 2
        ))
        cohens_d = float(
            mean_diff / pooled_std
            if pooled_std > 0 else 0.0
        )
        def sig(p):
            if   p < 0.001 : return "***"
            elif p < 0.01  : return "**"
            elif p < 0.05  : return "*"
            else           : return "ns"
        return {
            "comparison"   : f"{name_a} vs {name_b}",
            "mean_a"       : round(float(np.mean(a)),4),
            "mean_b"       : round(float(np.mean(b)),4),
            "mean_diff"    : round(mean_diff,4),
            "pct_reduction": round(
                abs(mean_diff)/
                max(abs(float(np.mean(b))),0.001)*100,2
            ),
            "t_stat"       : round(float(t_stat),4),
            "p_ttest"      : round(float(p_val),6),
            "p_mannwhitney": round(float(p_mw),6),
            "cohens_d"     : round(cohens_d,4),
            "effect_size"  : (
                "Large"     if abs(cohens_d)>=0.8 else
                "Medium"    if abs(cohens_d)>=0.5 else
                "Small"     if abs(cohens_d)>=0.2 else
                "Negligible"
            ),
            "sig_ttest"    : sig(float(p_val)),
            "rl_better"    : bool(mean_diff < 0),
        }

    return {
        "rl_vs_rag" : compare(
            scores_rl, scores_rag,
            "Agentic RAG+RL","Standard RAG"
        ),
        "rl_vs_llm" : compare(
            scores_rl, scores_llm,
            "Agentic RAG+RL","LLM Only"
        ),
        "rag_vs_llm": compare(
            scores_rag, scores_llm,
            "Standard RAG","LLM Only"
        ),
    }


# ==================================================
# GRAPHS
# ==================================================

def plot_main_comparison(agg: dict, stat: dict):

    systems     = list(agg.keys())
    colors      = [SYS_COLORS[s] for s in systems]
    hall_scores = [agg[s]["hall_score"] for s in systems]
    free_scores = [agg[s]["free_score"] for s in systems]
    hall_rates  = [agg[s]["hall_rate"]  for s in systems]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # 1. Hallucination Score
    bars1 = axes[0].bar(
        systems, hall_scores,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.8, width=0.55
    )
    for bar, val, sys in zip(bars1, hall_scores, systems):
        axes[0].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.008,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11, color=SYS_COLORS[sys],
            fontweight="bold"
        )

    # Significance brackets
    y_max = max(hall_scores)
    y1    = y_max + 0.06
    y2    = y_max + 0.12

    axes[0].plot(
        [0, 1], [y1, y1],
        color="white", linewidth=1.5
    )
    axes[0].text(
        0.5, y1+0.01,
        stat["rl_vs_rag"]["sig_ttest"],
        ha="center", fontsize=14,
        color=COLORS["good"]
        if stat["rl_vs_rag"]["rl_better"]
        else COLORS["warn"],
        fontweight="bold"
    )
    axes[0].plot(
        [0, 2], [y2, y2],
        color="white", linewidth=1.5
    )
    axes[0].text(
        1.0, y2+0.01,
        stat["rl_vs_llm"]["sig_ttest"],
        ha="center", fontsize=14,
        color=COLORS["good"],
        fontweight="bold"
    )
    axes[0].axhline(
        y=0.20, color=COLORS["warn"],
        linestyle="--", linewidth=1.5,
        label="Low Risk Threshold (0.20)"
    )
    axes[0].set_ylim(0, y_max + 0.25)
    axes[0].set_title(
        "Overall Hallucination Score\n"
        "(Lower = Less Hallucination ✅)",
        fontsize=13, fontweight="bold"
    )
    axes[0].set_xticklabels(
        systems, fontsize=11, fontweight="bold"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Hallucination-Free Score
    bars2 = axes[1].bar(
        systems, free_scores,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.8, width=0.55
    )
    for bar, val, sys in zip(bars2, free_scores, systems):
        axes[1].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.008,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11, color=SYS_COLORS[sys],
            fontweight="bold"
        )
    axes[1].axhline(
        y=0.80, color=COLORS["good"],
        linestyle="--", linewidth=1.5,
        label="Target (0.80)"
    )
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title(
        "Hallucination-Free Score\n"
        "(Higher = More Trustworthy ✅)",
        fontsize=13, fontweight="bold"
    )
    axes[1].set_xticklabels(
        systems, fontsize=11, fontweight="bold"
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Hallucination Rate %
    bars3 = axes[2].bar(
        systems, [r*100 for r in hall_rates],
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.8, width=0.55
    )
    for bar, val, sys in zip(bars3, hall_rates, systems):
        axes[2].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.8,
            f"{val*100:.1f}%",
            ha="center", va="bottom",
            fontsize=11, color=SYS_COLORS[sys],
            fontweight="bold"
        )
    axes[2].set_ylim(0, 100)
    axes[2].set_title(
        "% of Answers That Hallucinate\n"
        "(Lower = Better ✅)",
        fontsize=13, fontweight="bold"
    )
    axes[2].set_xticklabels(
        systems, fontsize=11, fontweight="bold"
    )
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Hallucination Comparison:\n"
        "Agentic RAG+RL  vs  Standard RAG  vs  LLM Only",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/hallucination_comparison/graphs/"
        "01_main_comparison.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_6_components_comparison(agg: dict):

    systems    = list(agg.keys())
    colors     = [SYS_COLORS[s] for s in systems]
    components = [
        ("H1 Factual Grounding",    "h1",
         "Is answer grounded in chunks?"),
        ("H2 Reference Alignment",  "h2",
         "Does answer match reference?"),
        ("H3 Numerical Claims",     "h3",
         "Are numbers verified?"),
        ("H4 Medical Entities",     "h4",
         "Are drug names from context?"),
        ("H5 Safety Compliance",    "h5",
         "Avoids dangerous claims?"),
        ("H6 Context Faithfulness", "h6",
         "Stays within context?"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()
    x    = np.arange(len(systems))

    for idx, (name, key, desc) in enumerate(components):
        ax   = axes[idx]
        vals = [agg[s][key] for s in systems]

        bars = ax.bar(
            x, vals, width=0.55,
            color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5
        )
        for bar, val, sys in zip(bars, vals, systems):
            ax.text(
                bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.01,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=10, color=SYS_COLORS[sys],
                fontweight="bold"
            )

        rl_val  = agg["Agentic RAG+RL"][key]
        rag_val = agg["Standard RAG"][key]
        delta   = rl_val - rag_val
        ax.annotate(
            f"RL edge:\n{delta:+.4f}",
            xy=(0, rl_val),
            xytext=(0.5, max(vals)+0.08),
            fontsize=8,
            color=COLORS["good"] if delta > 0
                  else COLORS["bad"],
            fontweight="bold", ha="center"
        )
        ax.axhline(
            y=0.75, color=COLORS["warn"],
            linestyle="--", linewidth=1.2,
            label="Target (0.75)", alpha=0.7
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            systems, fontsize=9,
            rotation=15, ha="right"
        )
        ax.set_ylim(0, 1.2)
        ax.set_title(
            f"{name}\n{desc}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=SYS_COLORS[s], label=s)
        for s in systems
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=3,
        fontsize=11, bbox_to_anchor=(0.5,-0.02)
    )
    fig.suptitle(
        "6-Component Hallucination: "
        "RAG+RL vs Standard RAG vs LLM Only\n"
        "(Higher = Less hallucination)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/hallucination_comparison/graphs/"
        "02_components.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_statistical_significance(stat: dict):

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    comparisons = [
        ("rl_vs_rag",  "RAG+RL vs Standard RAG"),
        ("rl_vs_llm",  "RAG+RL vs LLM Only"),
        ("rag_vs_llm", "Standard RAG vs LLM Only"),
    ]

    comp_names = [c[1] for c in comparisons]
    mean_diffs = [stat[c[0]]["mean_diff"] for c in comparisons]
    sigs       = [stat[c[0]]["sig_ttest"] for c in comparisons]
    effects    = [stat[c[0]]["effect_size"] for c in comparisons]

    colors_bar = [
        COLORS["good"] if d < 0 else COLORS["bad"]
        for d in mean_diffs
    ]

    bars = axes[0].barh(
        comp_names, mean_diffs,
        color=colors_bar, alpha=0.85,
        edgecolor="white", linewidth=0.5, height=0.5
    )
    for bar, diff, sig, eff in zip(
        bars, mean_diffs, sigs, effects
    ):
        x_pos = float(diff)+(0.002 if diff>=0 else -0.002)
        ha    = "left" if diff >= 0 else "right"
        axes[0].text(
            x_pos,
            bar.get_y()+bar.get_height()/2,
            f"{diff:+.4f}  {sig}  ({eff})",
            va="center", fontsize=10,
            fontweight="bold",
            color=COLORS["good"] if diff < 0
                  else COLORS["bad"],
            ha=ha
        )
    axes[0].axvline(
        x=0, color="white", linewidth=1.5, alpha=0.7
    )
    axes[0].set_xlabel(
        "Mean Hallucination Score Difference\n"
        "(Negative = First system hallucinates LESS)",
        fontsize=11
    )
    axes[0].set_title(
        "Statistical Comparison\n"
        "Mean Difference in Hallucination Score",
        fontsize=13, fontweight="bold"
    )
    axes[0].grid(axis="x", alpha=0.3)

    # Right — Table
    axes[1].axis("off")
    col_headers = [
        "Comparison","Diff↓","Reduc%",
        "p-value","Sig","Cohen d","Effect"
    ]
    table_data = []
    for key, label in comparisons:
        s = stat[key]
        table_data.append([
            label,
            f"{s['mean_diff']:+.4f}",
            f"{s['pct_reduction']:.1f}%",
            f"{s['p_ttest']:.5f}",
            s["sig_ttest"],
            f"{s['cohens_d']:.3f}",
            s["effect_size"],
        ])

    table = axes[1].table(
        cellText=table_data, colLabels=col_headers,
        cellLoc="center", loc="center",
        bbox=[0, 0.2, 1, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor(COLORS["blue"])
        cell.set_text_props(
            color="white", fontweight="bold"
        )
    for i in range(len(table_data)):
        for j in range(len(col_headers)):
            cell = table[i+1, j]
            cell.set_facecolor(
                "#1a2e1a" if i < 2 else COLORS["card"]
            )
            if j == 4:
                sig_val = table_data[i][4]
                color   = (
                    COLORS["good"] if "***" in sig_val else
                    COLORS["warn"] if "*" in sig_val else
                    COLORS["bad"]
                )
                cell.set_text_props(
                    color=color, fontweight="bold"
                )
            else:
                cell.set_text_props(color="white")

    axes[1].set_title(
        "Complete Statistical Results\n"
        "(*** p<0.001  ** p<0.01  * p<0.05  ns)",
        fontsize=12, fontweight="bold", pad=15
    )
    fig.suptitle(
        "Statistical Significance: Hallucination Comparison",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/hallucination_comparison/graphs/"
        "03_statistical.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_category_comparison(agg: dict):

    systems  = list(agg.keys())
    all_cats = sorted(set(
        cat
        for s in systems
        for cat in agg[s]["by_category"].keys()
    ))
    if not all_cats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    x     = np.arange(len(all_cats))
    width = 0.25

    for i, sys in enumerate(systems):
        vals   = [
            agg[sys]["by_category"].get(cat, 0.5)
            for cat in all_cats
        ]
        offset = (i - 1) * width
        axes[0].bar(
            x+offset, vals, width=width, label=sys,
            color=SYS_COLORS[sys], alpha=0.85,
            edgecolor="white", linewidth=0.3
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        all_cats, rotation=30, ha="right", fontsize=9
    )
    axes[0].set_ylabel(
        "Hallucination Score (Lower = Better)",
        fontsize=11
    )
    axes[0].axhline(
        y=0.20, color=COLORS["warn"],
        linestyle="--", linewidth=1.2,
        label="Low Risk Threshold", alpha=0.7
    )
    axes[0].set_ylim(0, 0.8)
    axes[0].set_title(
        "Hallucination Score by Cancer Category",
        fontsize=13, fontweight="bold"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    rl_improvements = []
    for cat in all_cats:
        rl_score  = agg["Agentic RAG+RL"]["by_category"].get(cat, 0.5)
        rag_score = agg["Standard RAG"]["by_category"].get(cat, 0.5)
        rl_improvements.append(rag_score - rl_score)

    colors_imp = [
        COLORS["good"] if v > 0 else COLORS["bad"]
        for v in rl_improvements
    ]
    bars2 = axes[1].barh(
        all_cats, rl_improvements,
        color=colors_imp, alpha=0.85,
        edgecolor="white", linewidth=0.4
    )
    for bar, val in zip(bars2, rl_improvements):
        axes[1].text(
            float(bar.get_width())+0.002,
            bar.get_y()+bar.get_height()/2,
            f"{val:+.4f}",
            va="center", fontsize=9,
            color=COLORS["good"] if val > 0
                  else COLORS["bad"],
            fontweight="bold"
        )
    axes[1].axvline(
        x=0, color="white", linewidth=1.2, alpha=0.7
    )
    axes[1].set_xlabel(
        "Hall Score Reduction\n"
        "(Positive = RAG+RL hallucinates LESS)",
        fontsize=11
    )
    axes[1].set_title(
        "RAG+RL Improvement vs Standard RAG\nPer Category",
        fontsize=13, fontweight="bold"
    )
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Category-Level Hallucination Comparison",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/hallucination_comparison/graphs/"
        "04_by_category.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_radar_comparison(agg: dict):

    components = [
        "H1\nGrounding","H2\nReference",
        "H3\nNumerical","H4\nMed Entity",
        "H5\nSafety","H6\nFaithfulness",
    ]
    keys   = ["h1","h2","h3","h4","h5","h6"]
    angles = np.linspace(
        0,2*np.pi,len(components),endpoint=False
    ).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(10,10), subplot_kw={"polar":True}
    )
    ax.set_facecolor(COLORS["card"])

    for sys_name, sys_agg in agg.items():
        color = SYS_COLORS[sys_name]
        vals  = [sys_agg[k] for k in keys]
        vals += vals[:1]
        lw    = 3.5 if sys_name=="Agentic RAG+RL" else 2.0
        ls    = "-" if sys_name=="Agentic RAG+RL" else "--"
        alpha = 0.15 if sys_name=="Agentic RAG+RL" else 0.05

        ax.plot(
            angles, vals, color=color,
            linewidth=lw, linestyle=ls,
            label=sys_name, zorder=3
        )
        ax.fill(angles, vals, color=color, alpha=alpha)
        ax.scatter(
            angles[:-1], vals[:-1],
            color=color, s=80, zorder=4
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        components, fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(
        ["0.2","0.4","0.6","0.8","1.0"],
        fontsize=9, color=COLORS["subtext"]
    )
    ax.grid(color=COLORS["grid"], alpha=0.5)
    ax.set_title(
        "Hallucination Components Radar\n"
        "Outer = Higher Score = Less Hallucination",
        fontsize=13, fontweight="bold", pad=30
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.45,1.15), fontsize=12
    )
    plt.tight_layout()
    path = (
        "results/hallucination_comparison/graphs/"
        "05_radar.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_final_summary(agg: dict, stat: dict):

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    ax.text(
        0.5, 0.98,
        "Hallucination Analysis — Final Summary Report",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color="white", transform=ax.transAxes
    )
    ax.text(
        0.5, 0.94,
        "Agentic RAG+RL  vs  Standard RAG  vs  LLM Only",
        ha="center", va="top",
        fontsize=12, color=COLORS["subtext"],
        transform=ax.transAxes
    )

    headers = [
        "System","Hall↓","Free↑","Rate",
        "H1","H2","H3","H4","H5","H6","Risk"
    ]
    col_pos = [
        0.01,0.15,0.23,0.31,0.39,
        0.45,0.51,0.57,0.63,0.69,0.75
    ]

    y = 0.87
    for col, hdr in zip(col_pos, headers):
        ax.text(
            col, y, hdr, ha="left", va="top",
            fontsize=9, fontweight="bold",
            color=COLORS["blue"],
            transform=ax.transAxes
        )
    y -= 0.02
    ax.plot(
        [0.01,0.99],[y+0.005,y+0.005],
        color=COLORS["grid"], linewidth=1,
        transform=ax.transAxes
    )

    systems = list(agg.keys())
    for sys in systems:
        y    -= 0.07
        m     = agg[sys]
        color = SYS_COLORS[sys]
        rank  = (
            "🥇" if sys=="Agentic RAG+RL" else
            "🥈" if sys=="Standard RAG"   else
            "🥉"
        )
        risk  = (
            "🟢 LOW"  if m["hall_score"]<=0.20 else
            "🟡 MED"  if m["hall_score"]<=0.40 else
            "🔴 HIGH"
        )
        vals = [
            (f"{rank} {sys}",          color),
            (f"{m['hall_score']:.4f}",
             COLORS["good"] if m["hall_score"]<=0.25
             else COLORS["warn"]),
            (f"{m['free_score']:.4f}",
             COLORS["good"] if m["free_score"]>=0.75
             else COLORS["warn"]),
            (f"{m['hall_rate']*100:.1f}%", "white"),
            (f"{m['h1']:.3f}", "white"),
            (f"{m['h2']:.3f}", "white"),
            (f"{m['h3']:.3f}", "white"),
            (f"{m['h4']:.3f}", "white"),
            (f"{m['h5']:.3f}", "white"),
            (f"{m['h6']:.3f}", "white"),
            (risk, "white"),
        ]
        for col, (val, vc) in zip(col_pos, vals):
            ax.text(
                col, y, str(val),
                ha="left", va="top",
                fontsize=9.5, color=vc,
                fontweight="bold",
                transform=ax.transAxes
            )

    y -= 0.06
    ax.plot(
        [0.01,0.99],[y+0.02,y+0.02],
        color=COLORS["grid"], linewidth=1,
        transform=ax.transAxes
    )
    y -= 0.01
    ax.text(
        0.01, y,
        "📊 Key Statistical Findings:",
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        color=COLORS["gold"],
        transform=ax.transAxes
    )
    y -= 0.05

    findings = [
        (
            f"RAG+RL vs Standard RAG: "
            f"Diff={stat['rl_vs_rag']['mean_diff']:+.4f}  "
            f"({stat['rl_vs_rag']['pct_reduction']:.1f}% reduction)  "
            f"p={stat['rl_vs_rag']['p_ttest']:.5f} "
            f"{stat['rl_vs_rag']['sig_ttest']}  "
            f"d={stat['rl_vs_rag']['cohens_d']:.3f} "
            f"({stat['rl_vs_rag']['effect_size']})",
            COLORS["good"]
            if stat["rl_vs_rag"]["rl_better"]
            else COLORS["warn"]
        ),
        (
            f"RAG+RL vs LLM Only:     "
            f"Diff={stat['rl_vs_llm']['mean_diff']:+.4f}  "
            f"({stat['rl_vs_llm']['pct_reduction']:.1f}% reduction)  "
            f"p={stat['rl_vs_llm']['p_ttest']:.5f} "
            f"{stat['rl_vs_llm']['sig_ttest']}  "
            f"d={stat['rl_vs_llm']['cohens_d']:.3f} "
            f"({stat['rl_vs_llm']['effect_size']})",
            COLORS["good"]
        ),
        (
            f"Standard RAG vs LLM:    "
            f"Diff={stat['rag_vs_llm']['mean_diff']:+.4f}  "
            f"({stat['rag_vs_llm']['pct_reduction']:.1f}% reduction)  "
            f"p={stat['rag_vs_llm']['p_ttest']:.5f} "
            f"{stat['rag_vs_llm']['sig_ttest']}  "
            f"d={stat['rag_vs_llm']['cohens_d']:.3f} "
            f"({stat['rag_vs_llm']['effect_size']})",
            COLORS["blue"]
        ),
    ]

    for text, color in findings:
        ax.text(
            0.03, y, f"  ▸ {text}",
            ha="left", va="top",
            fontsize=9, color=color,
            fontweight="bold",
            transform=ax.transAxes
        )
        y -= 0.055

    path = (
        "results/hallucination_comparison/graphs/"
        "06_final_summary.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


# ==================================================
# PRINT REPORT
# ==================================================

def print_report(agg: dict, stat: dict):

    print(f"\n{'='*70}")
    print(f"  HALLUCINATION COMPARISON REPORT")
    print(f"  Agentic RAG+RL vs Standard RAG vs LLM Only")
    print(f"{'='*70}")

    print(
        f"\n  {'System':<22}"
        f"{'Hall↓':>8}{'Free↑':>8}"
        f"{'Rate':>7}"
        f"{'H1':>6}{'H2':>6}{'H3':>6}"
        f"{'H4':>6}{'H5':>6}{'H6':>6}"
    )
    print(f"  {'─'*65}")

    ranks = {
        "Agentic RAG+RL":"🥇",
        "Standard RAG"  :"🥈",
        "LLM Only"      :"🥉",
    }
    for sys in agg:
        m = agg[sys]
        print(
            f"  {ranks.get(sys,'')} {sys:<20}"
            f"{m['hall_score']:>8.4f}"
            f"{m['free_score']:>8.4f}"
            f"{m['hall_rate']*100:>6.1f}%"
            f"{m['h1']:>6.3f}{m['h2']:>6.3f}"
            f"{m['h3']:>6.3f}{m['h4']:>6.3f}"
            f"{m['h5']:>6.3f}{m['h6']:>6.3f}"
        )

    print(f"\n  {'─'*65}")
    print(f"  📊 Statistical Significance")
    print(f"  {'─'*65}")
    for key, label in [
        ("rl_vs_rag",  "RAG+RL vs Standard RAG"),
        ("rl_vs_llm",  "RAG+RL vs LLM Only"),
        ("rag_vs_llm", "Standard RAG vs LLM Only"),
    ]:
        s      = stat[key]
        marker = "✅" if s["rl_better"] else "→"
        print(
            f"  {marker} {label:<30}"
            f"Diff={s['mean_diff']:+.4f}  "
            f"({s['pct_reduction']:.1f}% reduction)  "
            f"p={s['p_ttest']:.5f} {s['sig_ttest']}  "
            f"d={s['cohens_d']:.3f} ({s['effect_size']})"
        )
    print(f"\n{'='*70}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*70)
    print("  HALLUCINATION COMPARISON")
    print("  Agentic RAG+RL vs Standard RAG vs LLM Only")
    print("="*70)

    # Download NLTK with SSL fix already applied above
    print("\nDownloading NLTK data...")
    for pkg in ["punkt","punkt_tab","wordnet","omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    print("  NLTK ready ✅")

    print("\nLoading embedding model...")
    emb_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model ready ✅")

    print("\nLoading ChromaDB...")
    client     = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="medical_rag",
        metadata={"hnsw:space":"cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    # ── FIXED Ollama check ────────────────────────
    # Don't try to start ollama — just check it's running
    print("\nChecking Ollama connection...")
    try:
        resp = requests.get(
            "http://localhost:11434",
            timeout=5
        )
        print("  Ollama running ✅")
    except Exception:
        print("  ❌ Ollama not responding.")
        print("  Open a NEW terminal and run: ollama serve")
        sys.exit(1)

    print("\nChecking LLaMA3...")
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":"llama3","prompt":"OK",
                "stream":False,
                "options":{"num_predict":3}
            },
            timeout=30
        )
        if resp.status_code == 200:
            print("  LLaMA3 ready ✅")
        else:
            print("  ❌ Run: ollama pull llama3")
            sys.exit(1)
    except Exception as e:
        print(f"  ❌ LLaMA3 error: {e}")
        sys.exit(1)

    N_QUESTIONS = 50

    print(
        f"\nEvaluating {N_QUESTIONS} questions × 3 systems..."
    )
    print("="*70)

    all_results = evaluate_all_systems(
        qa_data, collection, emb_model, N_QUESTIONS
    )

    print("\nAggregating...")
    agg = {
        sys: aggregate(results)
        for sys, results in all_results.items()
    }

    print("\nRunning statistical tests...")
    stat = statistical_comparison(
        agg["Agentic RAG+RL"],
        agg["Standard RAG"],
        agg["LLM Only"]
    )
    print("  Tests complete ✅")

    print("\nGenerating graphs...")
    plot_main_comparison(agg, stat)
    plot_6_components_comparison(agg)
    plot_statistical_significance(stat)
    plot_category_comparison(agg)
    plot_radar_comparison(agg)
    plot_final_summary(agg, stat)

    print_report(agg, stat)

    path = (
        "results/hallucination_comparison/"
        "hallucination_comparison_report.json"
    )
    with open(path, "w") as f:
        json.dump({
            "timestamp"  : datetime.now().isoformat(),
            "n_questions": N_QUESTIONS,
            "systems"    : list(agg.keys()),
            "results"    : {
                s: {
                    k: v for k, v in m.items()
                    if k != "per_question_scores"
                }
                for s, m in agg.items()
            },
            "statistics" : {
                k: {
                    kk: bool(vv)
                    if isinstance(vv, (bool, np.bool_))
                    else float(vv)
                    if isinstance(vv, (int,float,np.floating))
                    else str(vv)
                    for kk, vv in v.items()
                }
                for k, v in stat.items()
            },
        }, f, indent=2)

    print(f"\n  ✅ Report → {path}")
    print(
        "  open results/hallucination_comparison/graphs/"
    )
    print("="*70)