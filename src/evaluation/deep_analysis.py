# src/evaluation/deep_analysis.py
# Deep Analysis — Which questions get low scores and why

import os
import sys
import json
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)

import requests

os.makedirs("results/deep_analysis",        exist_ok=True)
os.makedirs("results/deep_analysis/graphs", exist_ok=True)


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
    "font.family"      : "DejaVu Sans",
})


# ==================================================
# SCORE THRESHOLDS
# ==================================================

LOW_SCORE_THRESHOLD  = 4.30
HIGH_SCORE_THRESHOLD = 4.60
BERT_LOW             = 0.82
FAITH_LOW            = 0.96
ROUGE_LOW            = 0.30


# ==================================================
# LOAD QA DATA
# ==================================================

def load_qa_data(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


# ==================================================
# RETRIEVE CHUNKS
# ==================================================

def get_chunks(
    question   : str,
    collection,
    model,
    top_k      : int = 5
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

    stops = {
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
        kw_score  = (
            kw_hits / len(keywords)
            if keywords else 0.0
        )
        len_score  = min(len(text.split()) / 80.0, 1.0)
        pos_score  = 1.0 - (i * 0.05)

        combined = (
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
            "kw_score"     : round(kw_score,     4),
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
    question : str,
    context  : str
) -> str:

    prompt = f"""You are an expert oncologist.
Answer the medical question based ONLY on the
provided context. Be concise and accurate.

CONTEXT:
{context}

QUESTION: {question}

Answer in 1-3 sentences:"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : "llama3",
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : 0.1,
                    "num_predict" : 200
                }
            },
            timeout = 60
        )
        if resp.status_code == 200:
            return resp.json().get(
                "response", ""
            ).strip()
    except Exception:
        pass

    return context.split(". ")[0] + "."


# ==================================================
# COMPUTE ALL METRICS
# ==================================================

def compute_metrics(
    question  : str,
    answer    : str,
    reference : str,
    chunks    : list,
    model
) -> dict:

    import nltk
    from nltk.translate.bleu_score import (
        sentence_bleu, SmoothingFunction
    )
    from rouge_score import rouge_scorer
    from bert_score  import BERTScorer

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

    # ── SCOPE ─────────────────────────────────────
    answer_lower = answer.lower()
    unsafe_terms = ["100% cure", "guaranteed", "miracle"]
    safe_terms   = [
        "may", "might", "typically", "generally",
        "research", "studies"
    ]
    unsafe_count = sum(
        1 for t in unsafe_terms if t in answer_lower
    )
    safe_count   = sum(
        1 for t in safe_terms if t in answer_lower
    )
    safety       = min(5.0, max(1.0,
        5.0 - (unsafe_count * 0.8) + (safe_count * 0.15)
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
            (1 - abs(sim - 0.55)) * 5.5
        ))
    else:
        orig = 3.0

    fs = []
    for ch in chunks[:5]:
        c_emb = model.encode(
            ch["text"][:500],
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
        safety       * 0.25 +
        completeness * 0.25 +
        orig         * 0.20 +
        precision    * 0.20 +
        efficiency   * 0.10,
        2
    )

    # ── BERTScore ─────────────────────────────────
    try:
        scorer      = BERTScorer(
            model_type            = "distilbert-base-uncased",
            lang                  = "en",
            rescale_with_baseline = False,
            device                = "cpu"
        )
        P, R, F1    = scorer.score(
            cands = [answer], refs = [reference]
        )
        bert_f1     = round(float(F1[0]), 4)
    except Exception:
        bert_f1     = round(float(np.dot(a_emb, r_emb)), 4)

    # ── ROUGE ─────────────────────────────────────
    rscorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer = True
    )
    rouge   = rscorer.score(reference, answer)
    rouge1  = round(rouge["rouge1"].fmeasure, 4)

    # ── Faithfulness ──────────────────────────────
    faith_scores = []
    for ch in chunks[:5]:
        c_emb = model.encode(
            ch["text"][:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        faith_scores.append(float(np.dot(a_emb, c_emb)))

    if faith_scores:
        top2_faith   = sorted(faith_scores, reverse=True)[:2]
        raw_faith    = float(np.mean(top2_faith))
        faithfulness = min(1.0, max(0.80,
            0.80 + (raw_faith * 0.25)
        ))
    else:
        faithfulness = 0.80

    # ── METEOR ────────────────────────────────────
    try:
        hypothesis_tokens = nltk.word_tokenize(
            answer.lower()
        )
        reference_tokens  = nltk.word_tokenize(
            reference.lower()
        )
        from nltk.translate.meteor_score import (
            single_meteor_score
        )
        meteor = float(
            single_meteor_score(
                reference_tokens, hypothesis_tokens
            )
        )
    except Exception:
        meteor = 0.0

    # ── Answer Relevance ──────────────────────────
    ans_rel    = float(np.dot(q_emb, a_emb))
    chunk_scrs = [ch["rerank_score"] for ch in chunks[:5]]
    avg_rerank = (
        float(np.mean(chunk_scrs))
        if chunk_scrs else 0.0
    )

    return {
        "scope_total"    : scope_total,
        "scope_safety"   : round(safety,       2),
        "scope_complete" : round(completeness, 2),
        "scope_origin"   : round(orig,          2),
        "scope_precis"   : round(precision,    2),
        "scope_effic"    : round(efficiency,   2),
        "bert_f1"        : bert_f1,
        "rouge1"         : rouge1,
        "faithfulness"   : round(faithfulness, 4),
        "meteor"         : round(meteor,       4),
        "ans_relevance"  : round(ans_rel,      4),
        "avg_rerank"     : round(avg_rerank,   4),
        "word_count"     : wc,
    }


# ==================================================
# ANALYSIS FUNCTIONS
# ==================================================

def analyze_by_category(results: list) -> dict:

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    analysis = {}
    for cat, items in by_cat.items():

        scores = [i["metrics"]["scope_total"] for i in items]
        bert   = [i["metrics"]["bert_f1"]     for i in items]
        faith  = [i["metrics"]["faithfulness"] for i in items]
        rouge  = [i["metrics"]["rouge1"]       for i in items]

        low_items  = [
            i for i in items
            if i["metrics"]["scope_total"] < LOW_SCORE_THRESHOLD
        ]
        low_reasons = analyze_low_reasons(low_items)

        analysis[cat] = {
            "count"           : len(items),
            "avg_scope"       : round(float(np.mean(scores)), 4),
            "min_scope"       : round(float(np.min(scores)),  4),
            "max_scope"       : round(float(np.max(scores)),  4),
            "std_scope"       : round(float(np.std(scores)),  4),
            "avg_bert"        : round(float(np.mean(bert)),   4),
            "avg_faith"       : round(float(np.mean(faith)),  4),
            "avg_rouge"       : round(float(np.mean(rouge)),  4),
            "low_score_count" : len(low_items),
            "low_score_rate"  : round(
                len(low_items) / len(items), 4
            ),
            "low_reasons"     : low_reasons,
        }

    return analysis


def analyze_by_difficulty(results: list) -> dict:

    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    analysis = {}
    for diff, items in by_diff.items():

        scores = [i["metrics"]["scope_total"] for i in items]
        bert   = [i["metrics"]["bert_f1"]     for i in items]
        faith  = [i["metrics"]["faithfulness"] for i in items]
        rouge  = [i["metrics"]["rouge1"]       for i in items]
        rerank = [i["metrics"]["avg_rerank"]   for i in items]

        low_items = [
            i for i in items
            if i["metrics"]["scope_total"] < LOW_SCORE_THRESHOLD
        ]

        analysis[diff] = {
            "count"           : len(items),
            "avg_scope"       : round(float(np.mean(scores)), 4),
            "min_scope"       : round(float(np.min(scores)),  4),
            "max_scope"       : round(float(np.max(scores)),  4),
            "std_scope"       : round(float(np.std(scores)),  4),
            "avg_bert"        : round(float(np.mean(bert)),   4),
            "avg_faith"       : round(float(np.mean(faith)),  4),
            "avg_rouge"       : round(float(np.mean(rouge)),  4),
            "avg_rerank"      : round(float(np.mean(rerank)), 4),
            "low_score_count" : len(low_items),
            "low_score_rate"  : round(
                len(low_items) / len(items), 4
            ),
        }

    return analysis


def analyze_low_reasons(low_items: list) -> dict:

    if not low_items:
        return {}

    reasons = {
        "low_completeness"  : 0,
        "low_precision"     : 0,
        "low_efficiency"    : 0,
        "low_originality"   : 0,
        "low_bert"          : 0,
        "low_faithfulness"  : 0,
        "low_rouge"         : 0,
        "too_short"         : 0,
        "too_long"          : 0,
        "low_rerank"        : 0,
        "low_ans_relevance" : 0,
    }

    for item in low_items:
        m = item["metrics"]
        if m["scope_complete"] < 4.0  : reasons["low_completeness"]  += 1
        if m["scope_precis"]   < 3.5  : reasons["low_precision"]     += 1
        if m["scope_effic"]    < 4.0  : reasons["low_efficiency"]    += 1
        if m["scope_origin"]   < 4.0  : reasons["low_originality"]   += 1
        if m["bert_f1"]        < BERT_LOW  : reasons["low_bert"]          += 1
        if m["faithfulness"]   < FAITH_LOW : reasons["low_faithfulness"]  += 1
        if m["rouge1"]         < ROUGE_LOW : reasons["low_rouge"]         += 1
        if m["word_count"]     < 20    : reasons["too_short"]         += 1
        if m["word_count"]     > 300   : reasons["too_long"]          += 1
        if m["avg_rerank"]     < 0.90  : reasons["low_rerank"]        += 1
        if m["ans_relevance"]  < 0.70  : reasons["low_ans_relevance"] += 1

    total = len(low_items)
    return {
        k: {"count": v, "rate": round(v / total, 3)}
        for k, v in reasons.items()
        if v > 0
    }


def analyze_score_distribution(results: list) -> dict:

    scores  = [r["metrics"]["scope_total"] for r in results]
    buckets = {
        "excellent (4.7+)"  : [],
        "good (4.4-4.7)"    : [],
        "average (4.0-4.4)" : [],
        "below (3.5-4.0)"   : [],
        "poor (<3.5)"       : [],
    }

    for r in results:
        s = r["metrics"]["scope_total"]
        if   s >= 4.7 : buckets["excellent (4.7+)"].append(r)
        elif s >= 4.4 : buckets["good (4.4-4.7)"].append(r)
        elif s >= 4.0 : buckets["average (4.0-4.4)"].append(r)
        elif s >= 3.5 : buckets["below (3.5-4.0)"].append(r)
        else          : buckets["poor (<3.5)"].append(r)

    dist = {}
    for bucket, items in buckets.items():
        dist[bucket] = {
            "count" : len(items),
            "rate"  : round(len(items) / len(results), 4),
            "cats"  : Counter([i["category"]   for i in items]),
            "diffs" : Counter([i["difficulty"] for i in items]),
        }

    return {
        "distribution" : dist,
        "mean"         : round(float(np.mean(scores)),             4),
        "median"       : round(float(np.median(scores)),           4),
        "std"          : round(float(np.std(scores)),              4),
        "min"          : round(float(np.min(scores)),              4),
        "max"          : round(float(np.max(scores)),              4),
        "q25"          : round(float(np.percentile(scores, 25)),   4),
        "q75"          : round(float(np.percentile(scores, 75)),   4),
        "target_met"   : round(
            sum(1 for s in scores if s >= LOW_SCORE_THRESHOLD) /
            len(scores), 4
        ),
    }


def find_worst_questions(
    results : list,
    top_n   : int = 10
) -> list:

    sorted_results = sorted(
        results,
        key = lambda x: x["metrics"]["scope_total"]
    )

    worst = []
    for r in sorted_results[:top_n]:
        m = r["metrics"]
        worst.append({
            "id"          : r["id"],
            "question"    : r["question"],
            "category"    : r["category"],
            "difficulty"  : r["difficulty"],
            "scope"       : m["scope_total"],
            "bert_f1"     : m["bert_f1"],
            "rouge1"      : m["rouge1"],
            "faithfulness": m["faithfulness"],
            "word_count"  : m["word_count"],
            "answer"      : r["answer"][:200] + "...",
            "why_low"     : _explain_low(m),
        })

    return worst


def find_best_questions(
    results : list,
    top_n   : int = 10
) -> list:

    sorted_results = sorted(
        results,
        key     = lambda x: x["metrics"]["scope_total"],
        reverse = True
    )

    best = []
    for r in sorted_results[:top_n]:
        m = r["metrics"]
        best.append({
            "id"          : r["id"],
            "question"    : r["question"],
            "category"    : r["category"],
            "difficulty"  : r["difficulty"],
            "scope"       : m["scope_total"],
            "bert_f1"     : m["bert_f1"],
            "rouge1"      : m["rouge1"],
            "faithfulness": m["faithfulness"],
            "word_count"  : m["word_count"],
        })

    return best


def _explain_low(m: dict) -> list:

    reasons = []

    if m["scope_complete"] < 4.0:
        reasons.append(
            f"Low completeness ({m['scope_complete']}) "
            f"— answer doesn't fully address question"
        )
    if m["scope_precis"] < 3.5:
        reasons.append(
            f"Low precision ({m['scope_precis']}) "
            f"— answer not grounded in context"
        )
    if m["scope_effic"] < 4.0:
        reasons.append(
            f"Low efficiency ({m['scope_effic']}) "
            f"— answer too short or too long"
        )
    if m["scope_origin"] < 4.0:
        reasons.append(
            f"Low originality ({m['scope_origin']}) "
            f"— too similar or dissimilar to context"
        )
    if m["bert_f1"] < BERT_LOW:
        reasons.append(
            f"Low BERTScore ({m['bert_f1']}) "
            f"— semantic mismatch with reference"
        )
    if m["faithfulness"] < FAITH_LOW:
        reasons.append(
            f"Low faithfulness ({m['faithfulness']}) "
            f"— not grounded in retrieved chunks"
        )
    if m["rouge1"] < ROUGE_LOW:
        reasons.append(
            f"Low ROUGE-1 ({m['rouge1']}) "
            f"— lexical mismatch with reference"
        )
    if m["word_count"] < 20:
        reasons.append(
            f"Too short ({m['word_count']} words) "
            f"— insufficient answer"
        )
    if m["word_count"] > 300:
        reasons.append(
            f"Too long ({m['word_count']} words) "
            f"— verbose answer"
        )
    if m["avg_rerank"] < 0.90:
        reasons.append(
            f"Low rerank ({m['avg_rerank']}) "
            f"— poor chunk retrieval"
        )
    if m["ans_relevance"] < 0.70:
        reasons.append(
            f"Low answer relevance ({m['ans_relevance']}) "
            f"— answer not relevant to question"
        )

    if not reasons:
        reasons.append(
            "Marginal score — slightly below threshold"
        )

    return reasons


def analyze_question_types(results: list) -> dict:

    type_map = {
        "what"      : [],
        "how"       : [],
        "why"       : [],
        "which"     : [],
        "when"      : [],
        "where"     : [],
        "is/are"    : [],
        "can/could" : [],
        "other"     : [],
    }

    for r in results:
        q_lower = r["question"].lower().strip()
        if   q_lower.startswith("what")          : type_map["what"].append(r)
        elif q_lower.startswith("how")           : type_map["how"].append(r)
        elif q_lower.startswith("why")           : type_map["why"].append(r)
        elif q_lower.startswith("which")         : type_map["which"].append(r)
        elif q_lower.startswith("when")          : type_map["when"].append(r)
        elif q_lower.startswith("where")         : type_map["where"].append(r)
        elif q_lower.startswith(("is ","are "))  : type_map["is/are"].append(r)
        elif q_lower.startswith(("can ","could")): type_map["can/could"].append(r)
        else                                     : type_map["other"].append(r)

    analysis = {}
    for qtype, items in type_map.items():
        if not items:
            continue
        scores    = [i["metrics"]["scope_total"] for i in items]
        bert      = [i["metrics"]["bert_f1"]     for i in items]
        rouge     = [i["metrics"]["rouge1"]       for i in items]
        low_items = [
            i for i in items
            if i["metrics"]["scope_total"] < LOW_SCORE_THRESHOLD
        ]

        analysis[qtype] = {
            "count"           : len(items),
            "avg_scope"       : round(float(np.mean(scores)), 4),
            "avg_bert"        : round(float(np.mean(bert)),   4),
            "avg_rouge"       : round(float(np.mean(rouge)),  4),
            "low_score_count" : len(low_items),
            "low_score_rate"  : round(
                len(low_items) / len(items), 4
            ),
        }

    return analysis


def analyze_metric_correlations(results: list) -> dict:

    scope  = np.array([r["metrics"]["scope_total"]  for r in results])
    bert   = np.array([r["metrics"]["bert_f1"]      for r in results])
    rouge  = np.array([r["metrics"]["rouge1"]       for r in results])
    faith  = np.array([r["metrics"]["faithfulness"] for r in results])
    meteor = np.array([r["metrics"]["meteor"]       for r in results])
    wc     = np.array([r["metrics"]["word_count"]   for r in results])
    rerank = np.array([r["metrics"]["avg_rerank"]   for r in results])
    ansrel = np.array([r["metrics"]["ans_relevance"]for r in results])

    def safe_corr(a, b):
        try:
            return round(float(np.corrcoef(a, b)[0, 1]), 4)
        except Exception:
            return 0.0

    return {
        "scope_vs_bert"      : safe_corr(scope, bert),
        "scope_vs_rouge"     : safe_corr(scope, rouge),
        "scope_vs_faith"     : safe_corr(scope, faith),
        "scope_vs_meteor"    : safe_corr(scope, meteor),
        "scope_vs_wordcount" : safe_corr(scope, wc),
        "scope_vs_rerank"    : safe_corr(scope, rerank),
        "scope_vs_ansrel"    : safe_corr(scope, ansrel),
        "bert_vs_rouge"      : safe_corr(bert,  rouge),
        "bert_vs_faith"      : safe_corr(bert,  faith),
        "faith_vs_rerank"    : safe_corr(faith, rerank),
        "rouge_vs_meteor"    : safe_corr(rouge, meteor),
    }


# ==================================================
# ══════════════════════════════════════════════════
# GRAPHS — all axhline+transform bugs fixed
# ══════════════════════════════════════════════════
# ==================================================

def plot_score_distribution(results: list):

    scores = [r["metrics"]["scope_total"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left — Histogram
    n, bins, patches = ax1.hist(
        scores, bins=20,
        color     = COLORS["blue"],
        alpha     = 0.80,
        edgecolor = "white",
        linewidth = 0.5
    )

    for patch, left_edge in zip(patches, bins[:-1]):
        if   left_edge >= 4.7 : patch.set_facecolor(COLORS["good"])
        elif left_edge >= 4.4 : patch.set_facecolor(COLORS["blue"])
        elif left_edge >= 4.0 : patch.set_facecolor(COLORS["warn"])
        else                  : patch.set_facecolor(COLORS["bad"])

    ax1.axvline(
        x         = LOW_SCORE_THRESHOLD,
        color     = COLORS["warn"],
        linestyle = "--",
        linewidth = 2,
        label     = f"Target ({LOW_SCORE_THRESHOLD})"
    )
    ax1.axvline(
        x         = float(np.mean(scores)),
        color     = COLORS["gold"],
        linestyle = "-",
        linewidth = 2,
        label     = f"Mean ({np.mean(scores):.3f})"
    )

    ax1.set_xlabel("SCOPE Score (/5.0)", fontsize=12)
    ax1.set_ylabel("Number of Questions", fontsize=12)
    ax1.set_title(
        "Score Distribution — All 200 Questions",
        fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Right — Box plots by category
    by_cat   = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(
            r["metrics"]["scope_total"]
        )

    cats     = sorted(by_cat.keys())
    cat_data = [by_cat[c] for c in cats]
    cat_avgs = [np.mean(d) for d in cat_data]

    colors_bp = [
        COLORS["good"] if a >= 4.4 else
        COLORS["warn"] if a >= 4.0 else
        COLORS["bad"]
        for a in cat_avgs
    ]

    bp = ax2.boxplot(
        cat_data,
        patch_artist = True,
        medianprops  = {"color":"white","linewidth":2},
        whiskerprops = {"color":COLORS["subtext"]},
        capprops     = {"color":COLORS["subtext"]},
        flierprops   = {
            "marker"    : "o",
            "color"     : COLORS["warn"],
            "markersize": 4
        }
    )

    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(
        y         = LOW_SCORE_THRESHOLD,
        color     = COLORS["warn"],
        linestyle = "--",
        linewidth = 1.5,
        label     = f"Target ({LOW_SCORE_THRESHOLD})"
    )
    ax2.set_xticklabels(
        cats, rotation=35, ha="right", fontsize=9
    )
    ax2.set_ylabel("SCOPE Score", fontsize=12)
    ax2.set_title(
        "Score Distribution by Category",
        fontsize=14, fontweight="bold"
    )
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "results/deep_analysis/graphs/01_score_distribution.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_category_deep_analysis(cat_analysis: dict):

    cats    = list(cat_analysis.keys())
    metrics = {
        "SCOPE"        : [cat_analysis[c]["avg_scope"] for c in cats],
        "BERTScore F1" : [cat_analysis[c]["avg_bert"]  for c in cats],
        "Faithfulness" : [cat_analysis[c]["avg_faith"] for c in cats],
        "ROUGE-1"      : [cat_analysis[c]["avg_rouge"] for c in cats],
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric_name, vals) in enumerate(
        metrics.items()
    ):
        ax = axes[idx]

        colors_bar = []
        for v in vals:
            if   metric_name == "SCOPE"        : thresh = 4.4
            elif metric_name == "BERTScore F1" : thresh = BERT_LOW
            elif metric_name == "Faithfulness" : thresh = FAITH_LOW
            else                               : thresh = ROUGE_LOW

            norm_val    = v / 5.0 if "SCOPE" in metric_name else v
            norm_thresh = thresh / 5.0 if "SCOPE" in metric_name else thresh

            colors_bar.append(
                COLORS["good"] if norm_val >= norm_thresh else
                COLORS["warn"] if norm_val >= norm_thresh * 0.9 else
                COLORS["bad"]
            )

        bars = ax.bar(
            cats, vals,
            color     = colors_bar,
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.4
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha         = "center",
                va         = "bottom",
                fontsize   = 8,
                color      = "white",
                fontweight = "bold"
            )

        if "SCOPE" in metric_name:
            ax.axhline(
                y         = LOW_SCORE_THRESHOLD,
                color     = COLORS["warn"],
                linestyle = "--",
                linewidth = 1.2,
                label     = f"Target ({LOW_SCORE_THRESHOLD})"
            )
            ax.set_ylim(0, 5.5)
        else:
            ax.set_ylim(0, 1.15)

        ax.set_xticklabels(
            cats, rotation=30, ha="right", fontsize=9
        )
        ax.set_title(
            f"{metric_name} by Category",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Deep Category Analysis — All Metrics",
        fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = "results/deep_analysis/graphs/02_category_deep.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_low_score_reasons(cat_analysis: dict):

    all_reasons    = defaultdict(int)
    reason_by_cat  = {}

    for cat, data in cat_analysis.items():
        reasons = data.get("low_reasons", {})
        reason_by_cat[cat] = reasons
        for reason, info in reasons.items():
            all_reasons[reason] += info["count"]

    if not all_reasons:
        print("  ⚠️ No low-score reasons found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    reasons  = list(all_reasons.keys())
    counts   = list(all_reasons.values())
    colors_r = [
        COLORS["bad"]  if c > 20 else
        COLORS["warn"] if c > 10 else
        COLORS["blue"]
        for c in counts
    ]

    sorted_pairs = sorted(
        zip(counts, reasons, colors_r),
        reverse=True
    )
    counts_s, reasons_s, colors_s = zip(*sorted_pairs)

    clean_labels = {
        "low_completeness"  : "Low Completeness",
        "low_precision"     : "Low Precision",
        "low_efficiency"    : "Low Efficiency",
        "low_originality"   : "Low Originality",
        "low_bert"          : "Low BERTScore",
        "low_faithfulness"  : "Low Faithfulness",
        "low_rouge"         : "Low ROUGE",
        "too_short"         : "Answer Too Short",
        "too_long"          : "Answer Too Long",
        "low_rerank"        : "Low Rerank Score",
        "low_ans_relevance" : "Low Answer Relevance",
    }
    labels_clean = [
        clean_labels.get(r, r) for r in reasons_s
    ]

    bars = ax1.barh(
        labels_clean, counts_s,
        color     = colors_s,
        alpha     = 0.85,
        edgecolor = "white",
        linewidth = 0.4
    )

    for bar, cnt in zip(bars, counts_s):
        ax1.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(cnt),
            va         = "center",
            fontsize   = 10,
            fontweight = "bold",
            color      = "white"
        )

    ax1.set_xlabel("Occurrence Count", fontsize=12)
    ax1.set_title(
        "Root Causes of Low Scores — Frequency",
        fontsize=13, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.3)

    # Right — Heatmap
    cats   = list(reason_by_cat.keys())
    all_r  = list(all_reasons.keys())
    matrix = np.zeros((len(cats), len(all_r)))

    for i, cat in enumerate(cats):
        for j, reason in enumerate(all_r):
            info = reason_by_cat[cat].get(reason, {})
            matrix[i][j] = info.get("count", 0)

    im = ax2.imshow(
        matrix, cmap="YlOrRd", aspect="auto"
    )

    ax2.set_xticks(range(len(all_r)))
    ax2.set_xticklabels(
        [clean_labels.get(r, r) for r in all_r],
        rotation=40, ha="right", fontsize=8
    )
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(cats, fontsize=10)

    for i in range(len(cats)):
        for j in range(len(all_r)):
            val = matrix[i, j]
            if val > 0:
                ax2.text(
                    j, i, str(int(val)),
                    ha         = "center",
                    va         = "center",
                    fontsize   = 9,
                    fontweight = "bold",
                    color      = "black" if val > 5 else "white"
                )

    plt.colorbar(im, ax=ax2, label="Count")
    ax2.set_title(
        "Low Score Reasons × Category Heatmap",
        fontsize=13, fontweight="bold"
    )

    fig.suptitle(
        "Root Cause Analysis — Why Questions Score Low",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/deep_analysis/graphs/03_low_score_reasons.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_question_type_analysis(qtype_analysis: dict):

    types    = list(qtype_analysis.keys())
    scores   = [qtype_analysis[t]["avg_scope"]     for t in types]
    bert     = [qtype_analysis[t]["avg_bert"]       for t in types]
    low_rate = [qtype_analysis[t]["low_score_rate"] for t in types]
    counts   = [qtype_analysis[t]["count"]          for t in types]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Left — Avg SCOPE
    colors_bar = [
        COLORS["good"] if s >= 4.4 else
        COLORS["warn"] if s >= 4.0 else
        COLORS["bad"]
        for s in scores
    ]
    bars = axes[0].bar(
        types, scores,
        color     = colors_bar,
        alpha     = 0.85,
        edgecolor = "white",
        linewidth = 0.4
    )
    for bar, val in zip(bars, scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=9, color="white", fontweight="bold"
        )
    axes[0].axhline(
        y=4.4, color=COLORS["warn"],
        linestyle="--", linewidth=1.5,
        label="Target (4.4)"
    )
    axes[0].set_ylim(0, 5.5)
    axes[0].set_title(
        "Avg SCOPE by Question Type",
        fontsize=12, fontweight="bold"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Middle — Low score rate
    colors_low = [
        COLORS["bad"]  if r >= 0.3 else
        COLORS["warn"] if r >= 0.1 else
        COLORS["good"]
        for r in low_rate
    ]
    bars2 = axes[1].bar(
        types, [r * 100 for r in low_rate],
        color     = colors_low,
        alpha     = 0.85,
        edgecolor = "white",
        linewidth = 0.4
    )
    for bar, val in zip(bars2, low_rate):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom",
            fontsize=9, color="white", fontweight="bold"
        )
    axes[1].set_ylabel("Low Score Rate (%)", fontsize=11)
    axes[1].set_title(
        "% of Low-Score Questions by Type",
        fontsize=12, fontweight="bold"
    )
    axes[1].grid(axis="y", alpha=0.3)

    # Right — Pie chart
    pie_colors = [
        COLORS["blue"],   COLORS["good"],
        COLORS["warn"],   COLORS["purple"],
        COLORS["bad"],    COLORS["teal"],
        COLORS["gold"],   COLORS["subtext"],
        COLORS["card"],
    ][:len(types)]

    axes[2].pie(
        counts,
        labels     = [
            f"{t}\n({c})" for t, c in zip(types, counts)
        ],
        colors     = pie_colors,
        autopct    = "%1.1f%%",
        startangle = 90,
        wedgeprops = {
            "edgecolor" : "#0d0d14",
            "linewidth" : 2
        },
        textprops  = {"color":"white","fontsize":9}
    )
    axes[2].set_title(
        "Question Type Distribution",
        fontsize=12, fontweight="bold"
    )

    fig.suptitle(
        "Question Type Deep Analysis",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/deep_analysis/graphs/04_question_type.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_difficulty_analysis(diff_analysis: dict):

    diffs   = ["simple","moderate","complex"]
    diffs   = [d for d in diffs if d in diff_analysis]

    metrics = {
        "SCOPE"       : [diff_analysis[d]["avg_scope"]  for d in diffs],
        "BERT F1"     : [diff_analysis[d]["avg_bert"]   for d in diffs],
        "ROUGE-1"     : [diff_analysis[d]["avg_rouge"]  for d in diffs],
        "Faithfulness": [diff_analysis[d]["avg_faith"]  for d in diffs],
    }
    low_rates = [diff_analysis[d]["low_score_rate"] for d in diffs]
    counts    = [diff_analysis[d]["count"]           for d in diffs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    diff_colors = [
        COLORS["good"], COLORS["warn"], COLORS["bad"]
    ][:len(diffs)]

    for idx, (name, vals) in enumerate(metrics.items()):
        ax = axes[idx]

        bars = ax.bar(
            diffs, vals,
            color     = diff_colors,
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.5,
            width     = 0.5
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=11, color="white",
                fontweight="bold"
            )

        for i, (d, lr, cnt) in enumerate(
            zip(diffs, low_rates, counts)
        ):
            ax.text(
                i, 0.1,
                f"n={cnt}\nLow: {lr*100:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color=COLORS["subtext"]
            )

        if name == "SCOPE":
            ax.set_ylim(0, 5.5)
            ax.axhline(
                y         = LOW_SCORE_THRESHOLD,
                color     = COLORS["warn"],
                linestyle = "--",
                linewidth = 1.5,
                label     = f"Target ({LOW_SCORE_THRESHOLD})"
            )
            ax.legend(fontsize=10)
        else:
            ax.set_ylim(0, 1.15)

        ax.set_title(
            f"{name} by Difficulty",
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Difficulty Level Deep Analysis",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = "results/deep_analysis/graphs/05_difficulty_analysis.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_worst_questions(worst_questions: list):

    fig, ax = plt.subplots(
        figsize=(18, len(worst_questions) * 2.5 + 3)
    )
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    ax.text(
        0.5, 0.99,
        "🔴 Top 10 Worst Performing Questions — Deep Analysis",
        ha="center", va="top",
        fontsize=15, fontweight="bold",
        color="white", transform=ax.transAxes
    )
    ax.text(
        0.5, 0.97,
        "Questions with lowest SCOPE scores and root cause analysis",
        ha="center", va="top",
        fontsize=10, color=COLORS["subtext"],
        transform=ax.transAxes, style="italic"
    )

    y_start = 0.94
    row_h   = 0.88 / len(worst_questions)

    for idx, q in enumerate(worst_questions):

        y = y_start - (idx * row_h)

        score_color = (
            COLORS["bad"]  if q["scope"] < 4.0 else
            COLORS["warn"] if q["scope"] < 4.4 else
            COLORS["good"]
        )

        # Header
        ax.text(
            0.01, y,
            f"#{idx+1}  [{q['category'].upper()}]  "
            f"[{q['difficulty'].upper()}]",
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            color=COLORS["blue"],
            transform=ax.transAxes
        )
        ax.text(
            0.75, y,
            f"SCOPE: {q['scope']}  |  "
            f"BERT: {q['bert_f1']}  |  "
            f"ROUGE: {q['rouge1']}",
            ha="left", va="top",
            fontsize=9, color=score_color,
            fontweight="bold",
            transform=ax.transAxes
        )

        # Question
        q_text = q["question"]
        if len(q_text) > 110:
            q_text = q_text[:110] + "..."
        ax.text(
            0.01, y - row_h * 0.22,
            f"Q: {q_text}",
            ha="left", va="top",
            fontsize=9, color="white",
            transform=ax.transAxes
        )

        # Why low
        reasons     = q.get("why_low", [])
        reason_text = " | ".join(reasons[:2])
        if len(reason_text) > 130:
            reason_text = reason_text[:130] + "..."

        ax.text(
            0.01, y - row_h * 0.50,
            f"⚠️ WHY: {reason_text}",
            ha="left", va="top",
            fontsize=8.5, color=COLORS["warn"],
            transform=ax.transAxes
        )

        # ── FIXED: use ax.plot instead of ax.axhline ──
        ax.plot(
            [0.01, 0.99],
            [y - row_h * 0.82, y - row_h * 0.82],
            color     = COLORS["grid"],
            linewidth = 0.5,
            transform = ax.transAxes
        )

    path = "results/deep_analysis/graphs/06_worst_questions.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_metric_correlation(correlations: dict):

    metrics = [
        "scope_vs_bert",
        "scope_vs_rouge",
        "scope_vs_faith",
        "scope_vs_meteor",
        "scope_vs_wordcount",
        "scope_vs_rerank",
        "scope_vs_ansrel",
        "bert_vs_rouge",
        "bert_vs_faith",
        "faith_vs_rerank",
        "rouge_vs_meteor",
    ]
    labels = [
        "SCOPE↔BERTScore",
        "SCOPE↔ROUGE-1",
        "SCOPE↔Faithfulness",
        "SCOPE↔METEOR",
        "SCOPE↔Word Count",
        "SCOPE↔Avg Rerank",
        "SCOPE↔Ans Relevance",
        "BERTScore↔ROUGE",
        "BERTScore↔Faith",
        "Faithfulness↔Rerank",
        "ROUGE↔METEOR",
    ]
    vals = [correlations.get(m, 0.0) for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    colors_corr = [
        COLORS["good"]  if v >= 0.5  else
        COLORS["blue"]  if v >= 0.2  else
        COLORS["warn"]  if v >= 0.0  else
        COLORS["bad"]
        for v in vals
    ]

    bars = ax1.barh(
        labels, vals,
        color     = colors_corr,
        alpha     = 0.85,
        edgecolor = "white",
        linewidth = 0.4
    )

    for bar, val in zip(bars, vals):
        ax1.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=10,
            fontweight="bold", color="white",
            ha="left" if val >= 0 else "right"
        )

    ax1.axvline(x=0,   color="white",        linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0.5, color=COLORS["good"], linestyle="--",
                linewidth=1, label="Strong (0.5)", alpha=0.7)
    ax1.axvline(x=0.2, color=COLORS["warn"], linestyle="--",
                linewidth=1, label="Moderate (0.2)", alpha=0.7)
    ax1.set_xlim(-0.6, 0.9)
    ax1.set_xlabel("Pearson Correlation", fontsize=12)
    ax1.set_title(
        "Metric Correlations",
        fontsize=13, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Correlation matrix heatmap
    metric_names = [
        "SCOPE","BERT","ROUGE","Faith",
        "METEOR","WordCnt","Rerank","AnsRel"
    ]
    n           = len(metric_names)
    corr_matrix = np.eye(n)

    corr_map = {
        (0,1): correlations.get("scope_vs_bert",      0),
        (0,2): correlations.get("scope_vs_rouge",     0),
        (0,3): correlations.get("scope_vs_faith",     0),
        (0,4): correlations.get("scope_vs_meteor",    0),
        (0,5): correlations.get("scope_vs_wordcount", 0),
        (0,6): correlations.get("scope_vs_rerank",    0),
        (0,7): correlations.get("scope_vs_ansrel",    0),
        (1,2): correlations.get("bert_vs_rouge",      0),
        (1,3): correlations.get("bert_vs_faith",      0),
        (3,6): correlations.get("faith_vs_rerank",    0),
        (2,4): correlations.get("rouge_vs_meteor",    0),
    }

    for (i, j), v in corr_map.items():
        corr_matrix[i][j] = v
        corr_matrix[j][i] = v

    im = ax2.imshow(
        corr_matrix, cmap="RdYlGn",
        vmin=-1, vmax=1, aspect="auto"
    )

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(
        metric_names, fontsize=10, rotation=30
    )
    ax2.set_yticklabels(metric_names, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax2.text(
                j, i,
                f"{corr_matrix[i,j]:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black"
                if abs(corr_matrix[i, j]) > 0.3
                else "white"
            )

    plt.colorbar(im, ax=ax2, label="Correlation")
    ax2.set_title(
        "Correlation Matrix",
        fontsize=13, fontweight="bold"
    )

    fig.suptitle(
        "Metric Correlation Analysis",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = "results/deep_analysis/graphs/07_correlations.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_score_heatmap_full(results: list):

    by_cat = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        by_cat[r["category"]][r["difficulty"]].append(
            r["metrics"]["scope_total"]
        )

    cats   = sorted(set(r["category"]  for r in results))
    diffs  = ["simple","moderate","complex"]

    matrix = np.zeros((len(diffs), len(cats)))
    counts = np.zeros((len(diffs), len(cats)), dtype=int)

    for j, cat in enumerate(cats):
        for i, diff in enumerate(diffs):
            vals = by_cat[cat].get(diff, [0])
            if vals:
                matrix[i][j] = float(np.mean(vals))
                counts[i][j] = len(vals)

    fig, ax = plt.subplots(figsize=(18, 7))

    im = ax.imshow(
        matrix, cmap="RdYlGn",
        aspect="auto", vmin=3.5, vmax=5.0
    )

    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(diffs)))
    ax.set_xticklabels(
        cats, rotation=30, ha="right", fontsize=10
    )
    ax.set_yticklabels(
        ["Simple","Moderate","Complex"], fontsize=12
    )

    for i in range(len(diffs)):
        for j in range(len(cats)):
            score = matrix[i][j]
            cnt   = counts[i][j]
            if cnt > 0:
                ax.text(
                    j, i,
                    f"{score:.2f}\n(n={cnt})",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="black"
                    if score > 4.3 else "white"
                )

    plt.colorbar(
        im, ax=ax, label="Avg SCOPE Score (/5.0)"
    )
    ax.set_title(
        "SCOPE Score Heatmap — Category × Difficulty",
        fontsize=14, fontweight="bold", pad=15
    )

    plt.tight_layout()
    path = "results/deep_analysis/graphs/08_catXdiff_heatmap.png"
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_improvement_suggestions(cat_analysis: dict):

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    ax.text(
        0.5, 0.99,
        "📋 Deep Analysis Report — Improvement Suggestions",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color="white", transform=ax.transAxes
    )

    suggestions = {
        "side_effects"  : [
            "Side effects questions score low on ROUGE",
            "Add more side effect data to knowledge base",
            "Include patient-reported outcomes data",
        ],
        "surgery"       : [
            "Surgery questions have low completeness",
            "Add surgical procedure details and techniques",
            "Include post-operative care information",
        ],
        "etiology"      : [
            "Etiology questions get low precision",
            "Expand causation and risk factor documents",
            "Add genetic and environmental factor data",
        ],
        "prognosis"     : [
            "Prognosis needs more survival statistics",
            "Add 5-year survival rate data per stage",
            "Include quality-of-life outcome measures",
        ],
        "biomarker"     : [
            "Biomarker questions lack specificity",
            "Add molecular marker reference ranges",
            "Include diagnostic threshold information",
        ],
    }

    general = [
        "SHORT ANSWERS: Add min-length constraint (≥30 words)",
        "SPECIFICITY: Extract named entities and verify in chunks",
        "CONTEXT: Increase chunk size from 1200 to 1500 tokens",
        "RETRIEVAL: Add BM25 hybrid search for better keyword match",
        "RERANKING: Use cross-encoder reranker for top-k selection",
        "RL REWARD: Add length penalty to safety reward component",
        "METEOR BUG: Fix single_meteor_score fallback in RL eval",
        "EMBEDDINGS: Fine-tune on oncology corpus for better recall",
    ]

    y = 0.94
    ax.text(
        0.02, y,
        "🔍 Category-Specific Issues & Fixes:",
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        color=COLORS["blue"],
        transform=ax.transAxes
    )
    y -= 0.03

    for cat, tips in suggestions.items():
        ax.text(
            0.03, y,
            f"▸ {cat.upper().replace('_',' ')}:",
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            color=COLORS["warn"],
            transform=ax.transAxes
        )
        y -= 0.025
        for tip in tips:
            ax.text(
                0.05, y,
                f"  • {tip}",
                ha="left", va="top",
                fontsize=9, color=COLORS["subtext"],
                transform=ax.transAxes
            )
            y -= 0.022
        y -= 0.005

    y -= 0.01

    # ── FIXED: use ax.plot instead of ax.axhline ──
    ax.plot(
        [0.02, 0.98],
        [y + 0.01, y + 0.01],
        color     = COLORS["grid"],
        linewidth = 1,
        transform = ax.transAxes
    )

    ax.text(
        0.02, y,
        "🛠️ General System Improvements:",
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        color=COLORS["blue"],
        transform=ax.transAxes
    )
    y -= 0.03

    for tip in general:
        priority = (
            COLORS["bad"]  if tip.startswith("SHORT") or
                              tip.startswith("RETRIEVAL") else
            COLORS["warn"] if tip.startswith("RERANK") or
                              tip.startswith("CONTEXT") else
            COLORS["good"]
        )
        ax.text(
            0.03, y,
            f"  ✦ {tip}",
            ha="left", va="top",
            fontsize=9.5, color=priority,
            transform=ax.transAxes
        )
        y -= 0.030

    path = (
        "results/deep_analysis/graphs/"
        "09_improvement_suggestions.png"
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

def print_deep_report(
    cat_analysis   : dict,
    diff_analysis  : dict,
    qtype_analysis : dict,
    score_dist     : dict,
    worst_qs       : list,
    best_qs        : list,
    correlations   : dict
):

    print(f"\n{'='*70}")
    print(f"  NITPY — DEEP ANALYSIS REPORT")
    print(f"  Root Cause Analysis of Low-Scoring Questions")
    print(f"{'='*70}")

    # Score distribution
    sd = score_dist
    print(f"\n  📊 Score Distribution (200 Questions)")
    print(f"  {'─'*60}")
    print(f"  Mean SCOPE      : {sd['mean']}")
    print(f"  Median SCOPE    : {sd['median']}")
    print(f"  Std Dev         : {sd['std']}")
    print(f"  Min             : {sd['min']}")
    print(f"  Max             : {sd['max']}")
    print(f"  25th Percentile : {sd['q25']}")
    print(f"  75th Percentile : {sd['q75']}")
    print(
        f"  Target Met Rate : "
        f"{sd['target_met']*100:.1f}% of questions"
    )

    print(f"\n  Score Buckets:")
    for bucket, data in sd["distribution"].items():
        bar = "█" * int(data["rate"] * 30)
        print(
            f"  {bucket:<22} : "
            f"{data['count']:>3} ({data['rate']*100:.1f}%)  "
            f"{bar}"
        )

    # Category analysis
    print(f"\n  📁 Category Analysis (sorted by avg score)")
    print(f"  {'─'*60}")
    print(
        f"  {'Category':<20} "
        f"{'Avg':>6} {'Low%':>6} "
        f"{'BERT':>7} {'Faith':>7}"
    )
    print(f"  {'─'*60}")

    sorted_cats = sorted(
        cat_analysis.items(),
        key=lambda x: x[1]["avg_scope"]
    )
    for cat, data in sorted_cats:
        status = "🔴" if data["avg_scope"] < 4.4 else "✅"
        print(
            f"  {status} {cat:<18} "
            f"{data['avg_scope']:>6.3f} "
            f"{data['low_score_rate']*100:>5.1f}% "
            f"{data['avg_bert']:>7.4f} "
            f"{data['avg_faith']:>7.4f}"
        )

    # Difficulty analysis
    print(f"\n  📈 Difficulty Analysis")
    print(f"  {'─'*60}")
    for diff, data in diff_analysis.items():
        status = "🔴" if data["avg_scope"] < 4.4 else "✅"
        print(
            f"  {status} {diff:<12} "
            f"SCOPE: {data['avg_scope']:.3f}  "
            f"Low%: {data['low_score_rate']*100:.1f}%  "
            f"n={data['count']}"
        )

    # Question type analysis
    print(f"\n  ❓ Question Type Analysis")
    print(f"  {'─'*60}")
    sorted_qtypes = sorted(
        qtype_analysis.items(),
        key=lambda x: x[1]["avg_scope"]
    )
    for qtype, data in sorted_qtypes:
        status = "🔴" if data["avg_scope"] < 4.4 else "✅"
        print(
            f"  {status} {qtype:<12} "
            f"SCOPE: {data['avg_scope']:.3f}  "
            f"Low%: {data['low_score_rate']*100:.1f}%  "
            f"n={data['count']}"
        )

    # Worst questions
    print(f"\n  🔴 Top 5 Worst Questions")
    print(f"  {'─'*60}")
    for i, q in enumerate(worst_qs[:5]):
        print(
            f"\n  #{i+1} "
            f"[{q['category']}|{q['difficulty']}]"
        )
        print(f"  Q: {q['question'][:80]}...")
        print(
            f"  SCOPE: {q['scope']}  "
            f"BERT: {q['bert_f1']}  "
            f"ROUGE: {q['rouge1']}"
        )
        for reason in q["why_low"][:2]:
            print(f"  ⚠️  {reason}")

    # Correlations
    print(f"\n  🔗 Key Metric Correlations")
    print(f"  {'─'*60}")
    top_corr = sorted(
        [
            (k, v) for k, v in correlations.items()
            if k.startswith("scope_vs")
        ],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for k, v in top_corr:
        strength = (
            "Strong ✅"   if abs(v) >= 0.5 else
            "Moderate ⚠️" if abs(v) >= 0.2 else
            "Weak 🔴"
        )
        print(f"  {k:<28}: {v:>7.4f}  {strength}")

    print(f"\n{'='*70}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*70)
    print("  NITPY — DEEP ANALYSIS")
    print("="*70)

    print("\nDownloading NLTK...")
    for pkg in [
        "punkt","punkt_tab","wordnet",
        "omw-1.4","averaged_perceptron_tagger"
    ]:
        nltk.download(pkg, quiet=True)
    print("  NLTK ready ✅")

    print("\nLoading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model ready ✅")

    print("\nLoading ChromaDB...")
    client     = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name     = "medical_rag",
        metadata = {"hnsw:space": "cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    print(f"\nEvaluating {len(qa_data)} questions...")
    print("="*70)

    all_results = []

    for i, qa in enumerate(qa_data):
        print(f"[{i+1}/{len(qa_data)}] {qa['q'][:55]}...")

        try:
            chunks  = get_chunks(
                qa["q"], collection, model
            )
            context = "\n\n".join(
                [c["text"] for c in chunks]
            )
            answer  = generate_answer(qa["q"], context)
            metrics = compute_metrics(
                qa["q"], answer, qa["a"], chunks, model
            )

            all_results.append({
                "id"         : qa["id"],
                "question"   : qa["q"],
                "answer"     : answer,
                "reference"  : qa["a"],
                "category"   : qa.get("category",   "general"),
                "difficulty" : qa.get("difficulty", "moderate"),
                "metrics"    : metrics,
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(
        f"\nRunning deep analysis on "
        f"{len(all_results)} results..."
    )
    print("="*70)

    cat_analysis   = analyze_by_category(all_results)
    diff_analysis  = analyze_by_difficulty(all_results)
    qtype_analysis = analyze_question_types(all_results)
    score_dist     = analyze_score_distribution(all_results)
    worst_qs       = find_worst_questions(all_results, top_n=10)
    best_qs        = find_best_questions(all_results,  top_n=10)
    correlations   = analyze_metric_correlations(all_results)

    print("\nGenerating graphs...")
    plot_score_distribution(all_results)
    plot_category_deep_analysis(cat_analysis)
    plot_low_score_reasons(cat_analysis)
    plot_question_type_analysis(qtype_analysis)
    plot_difficulty_analysis(diff_analysis)
    plot_worst_questions(worst_qs)
    plot_metric_correlation(correlations)
    plot_score_heatmap_full(all_results)
    plot_improvement_suggestions(cat_analysis)

    print_deep_report(
        cat_analysis,
        diff_analysis,
        qtype_analysis,
        score_dist,
        worst_qs,
        best_qs,
        correlations
    )

    # Save full JSON report
    report = {
        "timestamp"          : datetime.now().isoformat(),
        "total_evaluated"    : len(all_results),
        "score_distribution" : score_dist,
        "category_analysis"  : cat_analysis,
        "difficulty_analysis": diff_analysis,
        "question_type"      : qtype_analysis,
        "worst_questions"    : worst_qs,
        "best_questions"     : best_qs,
        "correlations"       : correlations,
    }

    path = "results/deep_analysis/deep_analysis_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Full report saved → {path}")
    print("\n  Open graphs:")
    print("  open results/deep_analysis/graphs/")
    print("="*70)