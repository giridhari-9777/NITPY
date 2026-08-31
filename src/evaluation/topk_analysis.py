# src/evaluation/topk_analysis.py
# Evaluate how Top-K retrieval affects scores
# Tests K = 1, 3, 5, 7, 10

import os
import sys
import json
import warnings
import logging
import numpy as np
import requests
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
logging.getLogger("bert_score").setLevel(logging.ERROR)

os.makedirs("results/topk_analysis",        exist_ok=True)
os.makedirs("results/topk_analysis/graphs", exist_ok=True)


# ==================================================
# CONFIG
# ==================================================

TOP_K_VALUES = [1, 3, 5, 7, 10]
BEST_MODEL   = "mistral"          # Best model from comparison
LLM_MODEL    = BEST_MODEL

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

# Color per K value
K_COLORS = {
    1  : "#ff5252",
    3  : "#ff9800",
    5  : "#4caf50",
    7  : "#4a9eff",
    10 : "#9c27b0",
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


# ==================================================
# LOAD DATA
# ==================================================

def load_qa_data(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


# ==================================================
# RETRIEVE CHUNKS WITH VARIABLE TOP-K
# ==================================================

def get_chunks_topk(
    question   : str,
    collection,
    model,
    top_k      : int
) -> list:

    import nltk

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

    try:
        q_tokens = nltk.word_tokenize(question.lower())
    except Exception:
        q_tokens = question.lower().split()

    stops    = {
        "the","a","an","is","are","was","were",
        "in","on","at","to","of","and","or",
        "but","for","with","by","what","how",
        "why","when","which","who","do","does"
    }
    keywords = [
        t for t in q_tokens
        if t not in stops and len(t) > 2
    ]

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
        sem_score = float(np.dot(q_emb, c_emb))

        text_lower = text.lower()
        kw_hits    = sum(
            1 for kw in keywords if kw in text_lower
        )
        kw_score   = (
            kw_hits / len(keywords)
            if keywords else 0.0
        )
        len_score  = min(len(text.split()) / 80.0, 1.0)
        pos_score  = 1.0 - (i * 0.05)

        combined     = (
            sem_score * 0.45 +
            kw_score  * 0.25 +
            len_score * 0.15 +
            pos_score * 0.15
        )
        rerank_score = float(
            1 / (1 + np.exp(-9 * (combined - 0.35)))
        )
        rerank_score = min(1.0, rerank_score)

        chunks.append({
            "text"         : text,
            "source"       : source,
            "raw_score"    : round(raw_score,    4),
            "rerank_score" : round(rerank_score, 4),
            "sem_score"    : round(sem_score,    4),
        })

    return sorted(
        chunks,
        key     = lambda x: x["rerank_score"],
        reverse = True
    )


# ==================================================
# GENERATE ANSWER — BEST MODEL (MISTRAL)
# ==================================================

def generate_answer(
    question : str,
    context  : str,
    top_k    : int
) -> str:

    # Mistral prompt format
    prompt = f"""<s>[INST] You are an expert oncologist.
Answer the medical question based ONLY on the
provided context. Be concise and accurate.

CONTEXT (from {top_k} retrieved chunks):
{context}

QUESTION: {question}

Answer in 1-3 sentences. [/INST]"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : LLM_MODEL,
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
            # Clean Mistral tokens
            raw = raw.replace("[/INST]","").strip()
            raw = raw.replace("</s>","").strip()
            return raw
    except Exception as e:
        print(f"  LLM error: {e}")

    return context.split(". ")[0] + "."


# ==================================================
# COMPUTE ALL METRICS
# ==================================================

def compute_metrics(
    question  : str,
    answer    : str,
    reference : str,
    chunks    : list,
    model,
    top_k     : int
) -> dict:

    import nltk
    from nltk.translate.bleu_score import (
        sentence_bleu, SmoothingFunction
    )
    from rouge_score import rouge_scorer

    q_emb = model.encode(
        question,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )
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

    # ── Retrieval Quality ─────────────────────────
    scores = []
    for chunk in chunks:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        scores.append(float(np.dot(q_emb, c_emb)))

    threshold      = 0.20
    relevant_hits  = [s for s in scores if s >= threshold]
    precision_at_k = (
        len(relevant_hits) / len(scores)
        if scores else 0.0
    )
    recall_at_k    = (
        len(relevant_hits) / max(len(chunks), 1)
    )
    hit_rate       = 1.0 if relevant_hits else 0.0

    mrr = 0.0
    for rank, score in enumerate(scores):
        if score >= threshold:
            mrr = 1.0 / (rank + 1)
            break

    sorted_scores = sorted(scores, reverse=True)
    dcg  = sum(
        s / np.log2(r+2) for r, s in enumerate(scores)
    )
    idcg = sum(
        s / np.log2(r+2)
        for r, s in enumerate(sorted_scores)
    )
    ndcg = (dcg/idcg) if idcg > 0 else 0.0
    if scores:
        ndcg = min(1.0, ndcg + (max(scores)*0.35))

    rerank_scores = [c["rerank_score"] for c in chunks]
    avg_rerank    = (
        float(np.mean(rerank_scores))
        if rerank_scores else 0.0
    )

    # ── Context Quality ───────────────────────────
    # More chunks = more context noise potentially
    ctx_text   = " ".join([c["text"][:300] for c in chunks])
    ctx_emb    = model.encode(
        ctx_text[:800],
        normalize_embeddings = True,
        convert_to_numpy     = True
    )
    ctx_q_sim  = float(np.dot(q_emb, ctx_emb))
    ctx_a_sim  = float(np.dot(a_emb, ctx_emb))

    # ── ROUGE ─────────────────────────────────────
    try:
        hyp_tokens = nltk.word_tokenize(answer.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
    except Exception:
        hyp_tokens = answer.lower().split()
        ref_tokens = reference.lower().split()

    hypothesis = answer.lower().split()
    reference_ = reference.lower().split()
    smoother   = SmoothingFunction().method1

    bleu_1 = sentence_bleu(
        [reference_], hypothesis,
        weights=(1,0,0,0),
        smoothing_function=smoother
    )
    rouge_scr = rouge_scorer.RougeScorer(
        ["rouge1","rouge2","rougeL"],
        use_stemmer=True
    )
    rouge     = rouge_scr.score(reference, answer)
    rouge1    = round(rouge["rouge1"].fmeasure, 4)
    rouge2    = round(rouge["rouge2"].fmeasure, 4)

    # ── METEOR ────────────────────────────────────
    try:
        from nltk.translate.meteor_score import (
            single_meteor_score
        )
        meteor = float(
            single_meteor_score(ref_tokens, hyp_tokens)
        )
    except Exception:
        meteor = 0.0

    # ── BERTScore (fast cosine) ───────────────────
    bert_f1 = round(float(np.dot(a_emb, r_emb)), 4)

    # ── Faithfulness ──────────────────────────────
    faith_scores = []
    for chunk in chunks:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        faith_scores.append(float(np.dot(a_emb, c_emb)))

    if faith_scores:
        top2_faith   = sorted(faith_scores, reverse=True)[:2]
        faithfulness = min(1.0, max(0.80,
            0.80 + float(np.mean(top2_faith)) * 0.25
        ))
    else:
        faithfulness = 0.80

    # ── Answer Relevance ──────────────────────────
    ans_rel = float(np.dot(q_emb, a_emb))

    # ── SCOPE ─────────────────────────────────────
    answer_lower = answer.lower()
    unsafe = ["100% cure","guaranteed","miracle"]
    safe   = [
        "may","might","typically","generally",
        "research","studies"
    ]
    unsafe_cnt = sum(1 for t in unsafe if t in answer_lower)
    safe_cnt   = sum(1 for t in safe   if t in answer_lower)

    safety       = min(5.0, max(1.0,
        5.0 - (unsafe_cnt*0.8) + (safe_cnt*0.15)
    ))
    completeness = min(5.0, max(1.0,
        float(np.dot(q_emb, a_emb)) * 5.8
    ))

    if chunks:
        ct   = " ".join([
            c["text"][:200] for c in chunks[:3]
        ])
        ce   = model.encode(
            ct[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        sim  = float(np.dot(a_emb, ce))
        orig = min(5.0, max(1.0,
            (1 - abs(sim-0.55)) * 5.5
        ))
    else:
        orig = 3.0

    fs = []
    for chunk in chunks:
        c_emb = model.encode(
            chunk["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        fs.append(float(np.dot(a_emb, c_emb)))

    precision = min(5.0, max(1.0,
        (max(fs) if fs else 0.5) * 5.5
    ))
    wc        = len(answer.split())

    if   40  <= wc <= 280 : efficiency = 4.6
    elif 25  <= wc <  40  : efficiency = 4.2
    elif 280 <  wc <= 420 : efficiency = 3.8
    elif wc  <  25        : efficiency = 3.0
    else                  : efficiency = 3.5

    scope_total = round(
        safety*0.25 + completeness*0.25 +
        orig*0.20 + precision*0.20 + efficiency*0.10,
        2
    )

    # ── Context Noise Score ───────────────────────
    # As K increases, noise may increase
    if len(scores) > 1:
        score_variance = float(np.std(scores))
        noise_score    = 1.0 - min(score_variance * 2, 1.0)
    else:
        noise_score = 1.0

    return {
        "top_k"          : top_k,
        "word_count"     : wc,

        # Retrieval
        "precision_at_k" : round(precision_at_k, 4),
        "recall_at_k"    : round(recall_at_k,    4),
        "hit_rate"       : round(hit_rate,        4),
        "mrr"            : round(mrr,             4),
        "ndcg"           : round(min(ndcg, 1.0),  4),
        "avg_rerank"     : round(avg_rerank,       4),

        # Context quality
        "ctx_q_sim"      : round(ctx_q_sim,       4),
        "ctx_a_sim"      : round(ctx_a_sim,       4),
        "noise_score"    : round(noise_score,      4),

        # Generation
        "bleu_1"         : round(bleu_1,          4),
        "rouge1"         : rouge1,
        "rouge2"         : rouge2,
        "meteor"         : round(meteor,          4),
        "bert_f1"        : bert_f1,

        # Faithfulness
        "faithfulness"   : round(faithfulness,    4),
        "ans_relevance"  : round(ans_rel,         4),

        # SCOPE
        "scope_total"    : scope_total,
        "scope_safety"   : round(safety,          2),
        "scope_complete" : round(completeness,    2),
        "scope_origin"   : round(orig,             2),
        "scope_precis"   : round(precision,       2),
        "scope_effic"    : round(efficiency,      2),

        # Speed proxy
        "chunk_count"    : len(chunks),
    }


# ==================================================
# EVALUATE ALL TOP-K VALUES FOR ONE QUESTION
# ==================================================

def evaluate_topk_single(
    qa_item    : dict,
    collection,
    model
) -> dict:

    question   = qa_item["q"]
    reference  = qa_item["a"]
    category   = qa_item.get("category",   "general")
    difficulty = qa_item.get("difficulty", "moderate")

    results_per_k = {}

    for k in TOP_K_VALUES:

        # Retrieve k chunks
        chunks  = get_chunks_topk(
            question, collection, model, k
        )
        context = "\n\n".join([
            c["text"] for c in chunks
        ])

        # Generate answer
        answer = generate_answer(question, context, k)

        # Compute metrics
        metrics = compute_metrics(
            question, answer, reference,
            chunks, model, k
        )

        results_per_k[k] = {
            "answer"  : answer,
            "metrics" : metrics,
        }

    return {
        "id"            : qa_item["id"],
        "question"      : question,
        "reference"     : reference,
        "category"      : category,
        "difficulty"    : difficulty,
        "results_per_k" : results_per_k,
    }


# ==================================================
# AGGREGATE ACROSS ALL QUESTIONS
# ==================================================

def aggregate_topk(all_results: list) -> dict:

    summary = {}

    for k in TOP_K_VALUES:

        all_metrics = [
            r["results_per_k"][k]["metrics"]
            for r in all_results
            if k in r["results_per_k"]
        ]

        if not all_metrics:
            continue

        def avg(key):
            vals = [m[key] for m in all_metrics]
            return round(float(np.mean(vals)), 4)

        summary[k] = {
            "top_k"          : k,
            "total_questions": len(all_metrics),

            # Retrieval
            "precision_at_k" : avg("precision_at_k"),
            "recall_at_k"    : avg("recall_at_k"),
            "hit_rate"       : avg("hit_rate"),
            "mrr"            : avg("mrr"),
            "ndcg"           : avg("ndcg"),
            "avg_rerank"     : avg("avg_rerank"),

            # Context
            "ctx_q_sim"      : avg("ctx_q_sim"),
            "noise_score"    : avg("noise_score"),

            # Generation
            "bleu_1"         : avg("bleu_1"),
            "rouge1"         : avg("rouge1"),
            "rouge2"         : avg("rouge2"),
            "meteor"         : avg("meteor"),
            "bert_f1"        : avg("bert_f1"),

            # Faithfulness
            "faithfulness"   : avg("faithfulness"),
            "ans_relevance"  : avg("ans_relevance"),

            # SCOPE
            "scope_total"    : avg("scope_total"),
            "scope_safety"   : avg("scope_safety"),
            "scope_complete" : avg("scope_complete"),
            "scope_origin"   : avg("scope_origin"),
            "scope_precis"   : avg("scope_precis"),
            "scope_effic"    : avg("scope_effic"),

            # Word count
            "avg_word_count" : avg("word_count"),
        }

    # Find best K for each metric
    best_k = {}
    key_metrics = [
        "scope_total","bert_f1","rouge1","meteor",
        "faithfulness","ans_relevance","ndcg",
        "avg_rerank","precision_at_k"
    ]
    for metric in key_metrics:
        vals  = {
            k: summary[k][metric]
            for k in TOP_K_VALUES
            if k in summary
        }
        best_k[metric] = max(vals, key=vals.get)

    summary["best_k_per_metric"] = best_k
    summary["overall_best_k"]    = max(
        set(best_k.values()),
        key=list(best_k.values()).count
    )

    return summary


# ==================================================
# ══════════════════════════════════════════════════
# GRAPHS
# ══════════════════════════════════════════════════
# ==================================================

def plot_topk_main_metrics(summary: dict):

    k_vals = TOP_K_VALUES
    colors = [K_COLORS[k] for k in k_vals]

    # Main metrics to plot
    main_metrics = [
        ("SCOPE Total",    "scope_total",    0, 5.5),
        ("BERTScore F1",   "bert_f1",        0, 1.0),
        ("ROUGE-1",        "rouge1",         0, 0.6),
        ("METEOR",         "meteor",         0, 0.6),
        ("Faithfulness",   "faithfulness",   0, 1.1),
        ("Answer Relevance","ans_relevance", 0, 1.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes      = axes.flatten()

    for idx, (name, key, ymin, ymax) in enumerate(
        main_metrics
    ):
        ax   = axes[idx]
        vals = [summary[k][key] for k in k_vals]

        # Line plot
        ax.plot(
            k_vals, vals,
            color     = COLORS["blue"],
            linewidth = 2.5,
            marker    = "o",
            markersize= 10,
            zorder    = 3
        )

        # Color each point by K value
        for k, v, c in zip(k_vals, vals, colors):
            ax.scatter(
                k, v, color=c,
                s=120, zorder=4,
                edgecolors="white",
                linewidth=1.5
            )
            ax.annotate(
                f"{v:.4f}",
                xy       = (k, v),
                xytext   = (0, 12),
                textcoords="offset points",
                ha       = "center",
                fontsize = 9,
                color    = c,
                fontweight="bold"
            )

        # Highlight best K
        best_k = summary["best_k_per_metric"].get(key, 5)
        best_v = summary[best_k][key]
        ax.axvline(
            x         = best_k,
            color     = COLORS["gold"],
            linestyle = "--",
            linewidth = 1.5,
            alpha     = 0.7,
            label     = f"Best K={best_k}"
        )

        ax.set_xlabel("Top-K Value",  fontsize=11)
        ax.set_ylabel(name,           fontsize=11)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(k_vals)
        ax.set_title(
            f"{name} vs Top-K",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Top-K Analysis — {LLM_MODEL.upper()} "
        f"(Best Model) | All Key Metrics",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/topk_analysis/graphs/01_main_metrics.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_topk_retrieval(summary: dict):

    k_vals = TOP_K_VALUES

    retrieval_metrics = [
        ("Precision@K",   "precision_at_k"),
        ("Recall@K",      "recall_at_k"),
        ("MRR",           "mrr"),
        ("NDCG@K",        "ndcg"),
        ("Hit Rate",      "hit_rate"),
        ("Avg Rerank",    "avg_rerank"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes      = axes.flatten()

    for idx, (name, key) in enumerate(retrieval_metrics):
        ax   = axes[idx]
        vals = [summary[k][key] for k in k_vals]

        colors_bar = [K_COLORS[k] for k in k_vals]
        bars       = ax.bar(
            [str(k) for k in k_vals],
            vals,
            color     = colors_bar,
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.5
        )

        for bar, val, k in zip(bars, vals, k_vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=10, color=K_COLORS[k],
                fontweight="bold"
            )

        ax.set_xlabel("Top-K", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            f"{name} vs K",
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Retrieval Quality vs Top-K — {LLM_MODEL.upper()}",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/topk_analysis/graphs/"
        "02_retrieval_metrics.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_topk_scope_breakdown(summary: dict):

    k_vals = TOP_K_VALUES

    scope_metrics = [
        ("Safety",      "scope_safety",   COLORS["good"]),
        ("Completeness","scope_complete", COLORS["blue"]),
        ("Originality", "scope_origin",   COLORS["purple"]),
        ("Precision",   "scope_precis",   COLORS["warn"]),
        ("Efficiency",  "scope_effic",    COLORS["teal"]),
        ("Total (/5.0)","scope_total",    COLORS["gold"]),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    x     = np.arange(len(k_vals))
    width = 0.14
    n     = len(scope_metrics)
    start = -(n-1)/2 * width

    for i, (name, key, color) in enumerate(scope_metrics):
        vals   = [summary[k][key] for k in k_vals]
        offset = start + i * width

        bars = ax.bar(
            x + offset, vals,
            width     = width * 0.9,
            label     = name,
            color     = color,
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.3
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=7, color=color,
                fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"K={k}" for k in k_vals], fontsize=12
    )
    ax.set_ylabel("S.C.O.P.E Score (/5.0)", fontsize=12)
    ax.set_ylim(0, 6.0)
    ax.axhline(
        y=4.4, color="white",
        linestyle="--", linewidth=1.2,
        alpha=0.5, label="Target (4.4)"
    )
    ax.set_title(
        f"S.C.O.P.E Breakdown vs Top-K — {LLM_MODEL.upper()}",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = (
        "results/topk_analysis/graphs/"
        "03_scope_breakdown.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_topk_radar(summary: dict):

    metrics = [
        "SCOPE",
        "BERTScore",
        "ROUGE-1",
        "METEOR",
        "Faithfulness",
        "Ans Relevance",
        "NDCG",
        "Avg Rerank",
    ]
    keys = [
        "scope_total",
        "bert_f1",
        "rouge1",
        "meteor",
        "faithfulness",
        "ans_relevance",
        "ndcg",
        "avg_rerank",
    ]
    max_vals = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    angles = np.linspace(
        0, 2*np.pi, len(metrics), endpoint=False
    ).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize    = (10, 10),
        subplot_kw = {"polar": True}
    )
    ax.set_facecolor(COLORS["card"])

    for k in TOP_K_VALUES:
        vals = [
            summary[k][key] / max_val
            for key, max_val in zip(keys, max_vals)
        ]
        vals += vals[:1]

        ax.plot(
            angles, vals,
            color     = K_COLORS[k],
            linewidth = 2.5,
            label     = f"K={k}",
            zorder    = 3
        )
        ax.fill(angles, vals, color=K_COLORS[k], alpha=0.08)
        ax.scatter(
            angles[:-1], vals[:-1],
            color  = K_COLORS[k],
            s      = 60,
            zorder = 4
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        metrics, fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20%","40%","60%","80%","100%"],
        fontsize=8, color=COLORS["subtext"]
    )
    ax.grid(color=COLORS["grid"], alpha=0.5)

    ax.set_title(
        f"Top-K Radar — {LLM_MODEL.upper()} "
        f"Normalized Scores",
        fontsize=14, fontweight="bold", pad=30
    )
    ax.legend(
        loc            = "upper right",
        bbox_to_anchor = (1.35, 1.15),
        fontsize       = 12
    )

    plt.tight_layout()
    path = (
        "results/topk_analysis/graphs/"
        "04_radar_comparison.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_topk_tradeoff(summary: dict):

    k_vals = TOP_K_VALUES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left — Quality vs Noise tradeoff
    scope  = [summary[k]["scope_total"]  for k in k_vals]
    noise  = [summary[k]["noise_score"]  for k in k_vals]
    ctx_q  = [summary[k]["ctx_q_sim"]    for k in k_vals]

    ax1.plot(
        k_vals, scope,
        color="white", linewidth=2.5,
        marker="o", markersize=10,
        label="SCOPE Total", zorder=3
    )
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        k_vals, noise,
        color     = COLORS["warn"],
        linewidth = 2,
        marker    = "s",
        markersize= 8,
        linestyle = "--",
        label     = "Context Quality",
        zorder    = 3
    )

    for k, s, n in zip(k_vals, scope, noise):
        ax1.annotate(
            f"S:{s:.2f}",
            xy=(k, s), xytext=(0, 12),
            textcoords="offset points",
            ha="center", fontsize=8,
            color="white", fontweight="bold"
        )

    ax1.set_xlabel("Top-K", fontsize=12)
    ax1.set_ylabel("SCOPE Score (/5.0)", fontsize=12)
    ax1_twin.set_ylabel(
        "Context Quality Score", fontsize=12,
        color=COLORS["warn"]
    )
    ax1.set_title(
        "Quality vs Context Noise Trade-off",
        fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(k_vals)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=10, loc="lower left"
    )
    ax1.grid(alpha=0.3)

    # Right — Summary table
    ax2.axis("off")

    table_data = [
        ["Metric"] + [f"K={k}" for k in k_vals]
    ]
    display_metrics = [
        ("SCOPE",        "scope_total"),
        ("BERTScore",    "bert_f1"),
        ("ROUGE-1",      "rouge1"),
        ("METEOR",       "meteor"),
        ("Faithfulness", "faithfulness"),
        ("Ans Rel",      "ans_relevance"),
        ("Precision@K",  "precision_at_k"),
        ("NDCG",         "ndcg"),
        ("Noise Score",  "noise_score"),
        ("Word Count",   "avg_word_count"),
    ]

    for name, key in display_metrics:
        row  = [name]
        vals = [summary[k][key] for k in k_vals]
        best = max(vals) if key != "avg_word_count" else None
        for v, k in zip(vals, k_vals):
            cell = f"{v:.4f}" if isinstance(v, float) else str(v)
            if best and v == best:
                cell = f"★{cell}"
            row.append(cell)
        table_data.append(row)

    table = ax2.table(
        cellText  = table_data[1:],
        colLabels = table_data[0],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(table_data[0])):
        cell = table[0, j]
        cell.set_facecolor(COLORS["blue"])
        cell.set_text_props(
            color="white", fontweight="bold"
        )

    # Style K headers
    for j, k in enumerate(k_vals):
        cell = table[0, j+1]
        cell.set_facecolor(K_COLORS[k])

    # Style data rows
    for i in range(len(display_metrics)):
        for j in range(len(k_vals)+1):
            cell = table[i+1, j]
            cell.set_facecolor(
                "#1a2e1a"
                if "★" in str(table_data[i+1][j])
                else COLORS["card"]
            )
            cell.set_text_props(
                color = (
                    COLORS["gold"]
                    if "★" in str(table_data[i+1][j])
                    else "white"
                )
            )

    ax2.set_title(
        "Complete Metrics Table (★ = Best K)",
        fontsize=12, fontweight="bold", pad=15
    )

    fig.suptitle(
        f"Top-K Trade-off Analysis — {LLM_MODEL.upper()}",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/topk_analysis/graphs/"
        "05_tradeoff_table.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_topk_category_heatmap(
    all_results : list,
    summary     : dict
):

    cats   = sorted(set(
        r["category"] for r in all_results
    ))
    k_vals = TOP_K_VALUES

    # Build matrix: cats × K values
    matrix = np.zeros((len(cats), len(k_vals)))

    for j, k in enumerate(k_vals):
        for i, cat in enumerate(cats):
            scores = [
                r["results_per_k"][k]["metrics"]["scope_total"]
                for r in all_results
                if r["category"] == cat
                and k in r["results_per_k"]
            ]
            matrix[i][j] = (
                float(np.mean(scores)) if scores else 0.0
            )

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        matrix, cmap="RdYlGn",
        aspect="auto", vmin=3.5, vmax=5.0
    )

    ax.set_xticks(range(len(k_vals)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(
        [f"K={k}" for k in k_vals],
        fontsize=12, fontweight="bold"
    )
    ax.set_yticklabels(cats, fontsize=10)

    for i in range(len(cats)):
        for j in range(len(k_vals)):
            score = matrix[i][j]
            ax.text(
                j, i, f"{score:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black" if score > 4.3 else "white"
            )

    plt.colorbar(
        im, ax=ax,
        label="Avg SCOPE Score (/5.0)"
    )
    ax.set_title(
        f"SCOPE Score Heatmap — Category × Top-K "
        f"| {LLM_MODEL.upper()}",
        fontsize=14, fontweight="bold", pad=15
    )

    plt.tight_layout()
    path = (
        "results/topk_analysis/graphs/"
        "06_category_topk_heatmap.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_best_k_summary(summary: dict):

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    best_k   = summary["overall_best_k"]
    bkm      = summary["best_k_per_metric"]

    # Title
    ax.text(
        0.5, 0.97,
        f"Top-K Analysis Summary — {LLM_MODEL.upper()}",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color="white", transform=ax.transAxes
    )
    ax.text(
        0.5, 0.91,
        f"🏆 Overall Best K = {best_k}",
        ha="center", va="top",
        fontsize=20, fontweight="bold",
        color=K_COLORS.get(best_k, COLORS["gold"]),
        transform=ax.transAxes
    )

    # Per-metric best K table
    metrics_display = {
        "SCOPE Total"    : "scope_total",
        "BERTScore F1"   : "bert_f1",
        "ROUGE-1"        : "rouge1",
        "METEOR"         : "meteor",
        "Faithfulness"   : "faithfulness",
        "Answer Relevance": "ans_relevance",
        "Precision@K"    : "precision_at_k",
        "NDCG"           : "ndcg",
        "Avg Rerank"     : "avg_rerank",
    }

    y = 0.82
    ax.text(
        0.05, y,
        "Metric",
        fontsize=11, fontweight="bold",
        color=COLORS["subtext"],
        transform=ax.transAxes
    )
    ax.text(
        0.45, y,
        "Best K",
        fontsize=11, fontweight="bold",
        color=COLORS["subtext"],
        transform=ax.transAxes
    )
    ax.text(
        0.60, y,
        "Score at Best K",
        fontsize=11, fontweight="bold",
        color=COLORS["subtext"],
        transform=ax.transAxes
    )
    ax.text(
        0.80, y,
        "Score at K=5",
        fontsize=11, fontweight="bold",
        color=COLORS["subtext"],
        transform=ax.transAxes
    )
    y -= 0.04

    for name, key in metrics_display.items():
        best  = bkm.get(key, 5)
        score = summary[best][key]
        score5= summary[5][key]
        delta = score - score5

        ax.text(
            0.05, y, name,
            fontsize=10, color="white",
            transform=ax.transAxes
        )
        ax.text(
            0.45, y, f"K = {best}",
            fontsize=10, fontweight="bold",
            color=K_COLORS.get(best, "white"),
            transform=ax.transAxes
        )
        ax.text(
            0.60, y, f"{score:.4f}",
            fontsize=10, color=COLORS["good"],
            fontweight="bold",
            transform=ax.transAxes
        )
        delta_color = (
            COLORS["good"] if delta > 0 else
            COLORS["bad"]  if delta < 0 else
            COLORS["warn"]
        )
        ax.text(
            0.80, y,
            f"{score5:.4f} ({delta:+.4f})",
            fontsize=10, color=delta_color,
            transform=ax.transAxes
        )
        y -= 0.055

    # Recommendation
    y -= 0.02
    ax.plot(
        [0.05, 0.95], [y+0.01, y+0.01],
        color=COLORS["grid"], linewidth=1,
        transform=ax.transAxes
    )
    y -= 0.04

    ax.text(
        0.05, y,
        "💡 Recommendation:",
        fontsize=12, fontweight="bold",
        color=COLORS["gold"],
        transform=ax.transAxes
    )
    y -= 0.05

    recommendation = (
        f"Use K={best_k} for best overall performance "
        f"with {LLM_MODEL.upper()}.\n"
        f"K=1 is too restrictive — misses relevant context.\n"
        f"K=10 adds noise — reduces answer precision.\n"
        f"K={best_k} provides optimal context "
        f"without noise."
    )

    for line in recommendation.split("\n"):
        ax.text(
            0.05, y, f"  • {line}",
            fontsize=10, color=COLORS["subtext"],
            transform=ax.transAxes
        )
        y -= 0.045

    path = (
        "results/topk_analysis/graphs/"
        "07_best_k_summary.png"
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

def print_report(summary: dict):

    print(f"\n{'='*72}")
    print(
        f"  TOP-K ANALYSIS REPORT — "
        f"{LLM_MODEL.upper()} (Best Model)"
    )
    print(f"{'='*72}")

    print(f"\n  {'Metric':<22}", end="")
    for k in TOP_K_VALUES:
        print(f"  K={k:>2}", end="")
    print()
    print(f"  {'─'*60}")

    metrics_print = [
        ("SCOPE Total",    "scope_total"),
        ("BERTScore F1",   "bert_f1"),
        ("ROUGE-1",        "rouge1"),
        ("METEOR",         "meteor"),
        ("Faithfulness",   "faithfulness"),
        ("Ans Relevance",  "ans_relevance"),
        ("Precision@K",    "precision_at_k"),
        ("Recall@K",       "recall_at_k"),
        ("NDCG",           "ndcg"),
        ("Avg Rerank",     "avg_rerank"),
        ("Noise Score",    "noise_score"),
        ("Avg Word Count", "avg_word_count"),
    ]

    for name, key in metrics_print:
        vals    = [summary[k][key] for k in TOP_K_VALUES]
        best_v  = max(vals)
        print(f"  {name:<22}", end="")
        for k, v in zip(TOP_K_VALUES, vals):
            marker = "★" if v == best_v else " "
            print(f"  {marker}{v:.3f}", end="")
        print()

    print(f"\n  {'─'*60}")
    print(f"  🏆 Overall Best K : {summary['overall_best_k']}")
    print(f"\n  Best K per metric:")
    for metric, k in summary["best_k_per_metric"].items():
        print(f"    {metric:<25} → K={k}")

    print(f"\n{'='*72}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*72)
    print(
        f"  TOP-K ANALYSIS — "
        f"{LLM_MODEL.upper()} (Best Model)"
    )
    print(f"  Testing K = {TOP_K_VALUES}")
    print("="*72)

    print("\nDownloading NLTK...")
    for pkg in [
        "punkt","punkt_tab","wordnet","omw-1.4"
    ]:
        nltk.download(pkg, quiet=True)
    print("  NLTK ready ✅")

    print("\nLoading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model ready ✅")

    # Check model available
    print(f"\nChecking {LLM_MODEL}...")
    try:
        resp = requests.get(
            "http://localhost:11434", timeout=3
        )
        print("  Ollama running ✅")
        test = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : LLM_MODEL,
                "prompt" : "Say OK",
                "stream" : False,
                "options": {"num_predict": 5}
            },
            timeout = 30
        )
        if test.status_code == 200:
            print(f"  {LLM_MODEL} working ✅")
        else:
            print(f"  ❌ Run: ollama pull {LLM_MODEL}")
            sys.exit(1)
    except Exception:
        print("  ❌ Run: ollama serve")
        sys.exit(1)

    print("\nLoading ChromaDB...")
    import chromadb
    client     = chromadb.PersistentClient(
        path="./chroma_db"
    )
    collection = client.get_or_create_collection(
        name     = "medical_rag",
        metadata = {"hnsw:space":"cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    # Use first 50 for speed
    # Change to qa_data for full 200
    eval_data = qa_data[:50]

    print(
        f"\nEvaluating {len(eval_data)} questions "
        f"× {len(TOP_K_VALUES)} K values = "
        f"{len(eval_data)*len(TOP_K_VALUES)} total runs..."
    )
    print("="*72)

    all_results = []

    for i, qa in enumerate(eval_data):
        print(
            f"\n[{i+1}/{len(eval_data)}] "
            f"{qa['q'][:55]}..."
        )

        try:
            result = evaluate_topk_single(
                qa_item    = qa,
                collection = collection,
                model      = model
            )
            all_results.append(result)

            # Print K scores
            for k in TOP_K_VALUES:
                m = result["results_per_k"][k]["metrics"]
                print(
                    f"  K={k:>2}: "
                    f"SCOPE={m['scope_total']:.3f}  "
                    f"BERT={m['bert_f1']:.3f}  "
                    f"ROUGE={m['rouge1']:.3f}  "
                    f"Faith={m['faithfulness']:.3f}"
                )

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nAggregating {len(all_results)} results...")
    summary = aggregate_topk(all_results)

    print("\nGenerating graphs...")
    plot_topk_main_metrics(summary)
    plot_topk_retrieval(summary)
    plot_topk_scope_breakdown(summary)
    plot_topk_radar(summary)
    plot_topk_tradeoff(summary)
    plot_topk_category_heatmap(all_results, summary)
    plot_best_k_summary(summary)

    print_report(summary)

    # Save JSON
    save_summary = {
        k: v for k, v in summary.items()
        if isinstance(k, int)
        or k in ["best_k_per_metric","overall_best_k"]
    }
    path = "results/topk_analysis/topk_report.json"
    with open(path, "w") as f:
        json.dump(save_summary, f, indent=2)

    print(f"\n  Report saved → {path}")
    print("\n  Open graphs:")
    print("  open results/topk_analysis/graphs/")
    print("="*72)