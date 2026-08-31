# src/evaluation/self_correction_eval.py
# Evaluates Self-Correction Ability of the Agent
# FIXED VERSION — Better scoring + aggressive correction

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

os.makedirs("results/self_correction",        exist_ok=True)
os.makedirs("results/self_correction/graphs", exist_ok=True)


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


# ==================================================
# LOAD DATA
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
        raw_score    = 1 - result["distances"][0][i]
        text         = result["documents"][0][i]
        source       = result["metadatas"][0][i].get(
            "source", "unknown"
        )
        c_emb        = model.encode(
            text[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        rerank_score = float(np.dot(q_emb, c_emb))

        chunks.append({
            "text"         : text,
            "source"       : source,
            "raw_score"    : round(raw_score,    4),
            "rerank_score" : round(rerank_score, 4),
        })

    return sorted(
        chunks,
        key     = lambda x: x["rerank_score"],
        reverse = True
    )


# ==================================================
# STEP 1 — GENERATE INITIAL ANSWER
# ==================================================

def generate_initial_answer(
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

    return ""


# ==================================================
# STEP 2 — RULE-BASED SELF CRITIQUE
# Always runs — LLM critique is unreliable
# ==================================================

def self_critique(
    question       : str,
    initial_answer : str,
    context        : str,
    reference      : str,
    model
) -> dict:

    issues           = []
    quality_flags    = []

    a_lower = initial_answer.lower()
    q_lower = question.lower()

    # ── Check 1: Answer length ────────────────────
    wc = len(initial_answer.split())
    if wc < 15:
        issues.append(
            f"Too short ({wc} words — need ≥15)"
        )
        quality_flags.append("length")
    elif wc > 250:
        issues.append(
            f"Too verbose ({wc} words — keep ≤250)"
        )
        quality_flags.append("verbose")

    # ── Check 2: Medical terminology ─────────────
    medical_terms = [
        "cancer","tumor","treatment","therapy",
        "symptoms","diagnosis","stage","prognosis",
        "chemotherapy","radiation","surgery",
        "cells","oncology","biopsy","metastasis"
    ]
    med_count = sum(
        1 for t in medical_terms if t in a_lower
    )
    if med_count == 0:
        issues.append(
            "Missing medical terminology"
        )
        quality_flags.append("medical_terms")

    # ── Check 3: Question keyword coverage ───────
    stop_words = {
        "the","a","an","is","are","what","how",
        "why","of","in","for","does","do","to",
        "and","or","it","that","this","with","by"
    }
    q_words  = set(q_lower.split()) - stop_words
    a_words  = set(a_lower.split()) - stop_words
    coverage = (
        len(q_words & a_words) / max(len(q_words), 1)
    )
    if coverage < 0.25:
        issues.append(
            f"Low question coverage ({coverage:.2f})"
        )
        quality_flags.append("coverage")

    # ── Check 4: Semantic similarity to reference
    try:
        a_emb = model.encode(
            initial_answer[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        r_emb = model.encode(
            reference[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        ref_sim = float(np.dot(a_emb, r_emb))
        if ref_sim < 0.60:
            issues.append(
                f"Low reference match ({ref_sim:.3f})"
            )
            quality_flags.append("reference_sim")
    except Exception:
        ref_sim = 0.0

    # ── Check 5: Context grounding ────────────────
    try:
        q_emb = model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        q_a_sim = float(np.dot(q_emb, a_emb))
        if q_a_sim < 0.55:
            issues.append(
                f"Answer not relevant to question "
                f"({q_a_sim:.3f})"
            )
            quality_flags.append("relevance")
    except Exception:
        pass

    # ── Check 6: Vague answer ─────────────────────
    vague_phrases = [
        "it depends","i don't know","not sure",
        "cannot say","no information available",
        "i cannot answer","i'm not able"
    ]
    if any(p in a_lower for p in vague_phrases):
        issues.append("Answer is vague or evasive")
        quality_flags.append("vague")

    # ── Check 7: Unsafe claims ────────────────────
    unsafe = [
        "100% cure","guaranteed cure",
        "definitely cured","no side effects at all"
    ]
    if any(u in a_lower for u in unsafe):
        issues.append("Contains unsafe medical claim")
        quality_flags.append("unsafe")

    # ── Determine if correction needed ───────────
    # Correct if ANY issue found
    needs_correction = len(issues) > 0

    quality = (
        "POOR"       if len(issues) >= 3 else
        "INCOMPLETE" if len(issues) >= 1 else
        "GOOD"
    )

    return {
        "quality"          : quality,
        "issues"           : issues,
        "quality_flags"    : quality_flags,
        "needs_correction" : needs_correction,
        "n_issues"         : len(issues),
        "reason"           : (
            " | ".join(issues[:2])
            if issues else "Answer is satisfactory"
        ),
    }


# ==================================================
# STEP 3 — GENERATE CORRECTED ANSWER
# ==================================================

def generate_corrected_answer(
    question       : str,
    initial_answer : str,
    critique       : dict,
    context        : str
) -> str:

    issues_text = (
        "\n".join([f"- {i}" for i in critique["issues"]])
        if critique["issues"]
        else "- Answer needs to be more comprehensive"
    )

    correction_prompt = f"""You are an expert oncologist.
Your previous answer had quality issues.
Provide a BETTER, more complete answer.

QUESTION: {question}

YOUR PREVIOUS ANSWER (needs improvement):
{initial_answer}

ISSUES TO FIX:
{issues_text}

MEDICAL CONTEXT (ground truth — use this):
{context[:1000]}

Provide a CORRECTED answer that:
1. Fully addresses the question
2. Uses proper medical terminology
3. Is grounded in the context above
4. Is 2-4 sentences (30-150 words)
5. Is safe and evidence-based

CORRECTED ANSWER:"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : "llama3",
                "prompt" : correction_prompt,
                "stream" : False,
                "options": {
                    "temperature" : 0.2,
                    "num_predict" : 250
                }
            },
            timeout = 60
        )
        if resp.status_code == 200:
            raw = resp.json().get(
                "response", ""
            ).strip()
            # Remove prompt echoing if any
            if "CORRECTED ANSWER:" in raw:
                raw = raw.split(
                    "CORRECTED ANSWER:"
                )[-1].strip()
            return raw
    except Exception:
        pass

    return initial_answer


# ==================================================
# COMPUTE ANSWER QUALITY SCORE — IMPROVED
# ==================================================

def compute_answer_quality(
    question  : str,
    answer    : str,
    reference : str,
    chunks    : list,
    model
) -> float:

    if not answer or len(answer.strip()) < 5:
        return 0.0

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

    # 1. Semantic match with reference
    ref_sim = float(np.dot(a_emb, r_emb))

    # 2. Answer relevance to question
    q_sim   = float(np.dot(q_emb, a_emb))

    # 3. Faithfulness to chunks
    chunk_sims = []
    for chunk in chunks[:3]:
        c_emb = model.encode(
            chunk["text"][:400],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        chunk_sims.append(
            float(np.dot(a_emb, c_emb))
        )
    faith = (
        float(np.mean(chunk_sims))
        if chunk_sims else 0.0
    )

    # 4. Length score
    wc = len(answer.split())
    if   30 <= wc <= 150 : length_score = 1.0
    elif 15 <= wc <  30  : length_score = 0.7
    elif wc  <  15       : length_score = wc / 15.0
    else                 : length_score = 0.6

    # 5. Medical term coverage
    medical_terms = [
        "cancer","tumor","treatment","therapy",
        "symptoms","diagnosis","stage","prognosis",
        "cells","chemotherapy","radiation","surgery",
        "oncology","biopsy","metastasis","pathology"
    ]
    med_count     = sum(
        1 for t in medical_terms
        if t in answer.lower()
    )
    medical_score = min(1.0, med_count / 3.0)

    # 6. Question coverage
    stop_words = {
        "the","a","an","is","are","what","how",
        "why","of","in","for","does","do"
    }
    q_words   = set(question.lower().split()) - stop_words
    a_words   = set(answer.lower().split()) - stop_words
    coverage  = (
        len(q_words & a_words) / max(len(q_words), 1)
    )

    quality = round(
        ref_sim      * 0.30 +
        q_sim        * 0.20 +
        faith        * 0.20 +
        length_score * 0.15 +
        medical_score* 0.10 +
        coverage     * 0.05,
        4
    )

    return min(1.0, max(0.0, quality))


# ==================================================
# DETECT REAL ISSUES — IMPROVED
# ==================================================

def _detect_real_issues(
    answer    : str,
    reference : str,
    chunks    : list,
    model
) -> list:

    real_issues = []
    a_lower     = answer.lower()

    if len(answer.split()) < 15:
        real_issues.append("too_short")

    try:
        a_emb = model.encode(
            answer[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        r_emb = model.encode(
            reference[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        if float(np.dot(a_emb, r_emb)) < 0.60:
            real_issues.append("low_reference_match")
    except Exception:
        pass

    medical_terms = [
        "cancer","tumor","treatment","therapy",
        "diagnosis","prognosis","stage"
    ]
    if not any(t in a_lower for t in medical_terms):
        real_issues.append("missing_medical_terms")

    vague = ["it depends","not sure","cannot say"]
    if any(v in a_lower for v in vague):
        real_issues.append("vague_answer")

    if len(answer.split()) > 250:
        real_issues.append("too_verbose")

    return real_issues


# ==================================================
# EVALUATE SINGLE — SELF CORRECTION (IMPROVED)
# ==================================================

def evaluate_self_correction(
    qa_item    : dict,
    collection,
    model,
    max_iters  : int = 3
) -> dict:

    question   = qa_item["q"]
    reference  = qa_item["a"]
    category   = qa_item.get("category",   "general")
    difficulty = qa_item.get("difficulty", "moderate")

    chunks  = get_chunks(question, collection, model)
    context = "\n\n".join([c["text"] for c in chunks])

    # ── ITERATION 1 — Initial Answer ─────────────
    initial_answer  = generate_initial_answer(
        question, context
    )
    if not initial_answer:
        initial_answer = (
            "Unable to generate answer from context."
        )

    initial_quality = compute_answer_quality(
        question, initial_answer,
        reference, chunks, model
    )

    iterations = [{
        "iter"    : 1,
        "answer"  : initial_answer,
        "quality" : initial_quality,
        "critique": None,
        "type"    : "initial",
    }]

    current_answer  = initial_answer
    current_quality = initial_quality
    correction_made = False
    over_corrected  = False
    total_iters     = 1
    all_issues      = []
    all_flags       = []

    # ── ITERATIONS 2+ — Critique & Correct ───────
    for iter_num in range(2, max_iters + 1):

        # Run rule-based critique
        critique = self_critique(
            question, current_answer,
            context, reference, model
        )

        all_issues.extend(critique.get("issues", []))
        all_flags.extend(
            critique.get("quality_flags", [])
        )

        # Store critique on previous iteration
        iterations[-1]["critique"] = critique

        if not critique["needs_correction"]:
            # Quality is good — stop
            break

        # Generate corrected answer
        corrected = generate_corrected_answer(
            question, current_answer,
            critique, context
        )

        if not corrected or corrected == current_answer:
            break

        corrected_quality = compute_answer_quality(
            question, corrected,
            reference, chunks, model
        )

        iterations.append({
            "iter"    : iter_num,
            "answer"  : corrected,
            "quality" : corrected_quality,
            "critique": None,
            "type"    : "corrected",
        })

        total_iters = iter_num

        if corrected_quality > current_quality:
            correction_made = True
            current_answer  = corrected
            current_quality = corrected_quality
        else:
            # Got worse — mark and stop
            over_corrected = True
            break

        if current_quality >= 0.80:
            break

    # ── Final Metrics ─────────────────────────────
    final_quality    = current_quality
    quality_delta    = round(
        final_quality - initial_quality, 4
    )
    pct_improvement  = round(
        (quality_delta / max(initial_quality, 0.01))
        * 100, 2
    )

    # Error detection accuracy
    real_issues = _detect_real_issues(
        initial_answer, reference, chunks, model
    )
    n_real     = len(real_issues)
    n_detected = len(set(all_flags) & {
        "length","medical_terms","coverage",
        "reference_sim","relevance","vague","unsafe"
    })

    error_detection_acc = (
        round(n_detected / max(n_real, 1), 4)
        if n_real > 0
        else (1.0 if not all_flags else 0.8)
    )

    # Self-correction score — improved formula
    # Based on: did it improve? by how much?
    # + error detection + correction efficiency
    if correction_made:
        improvement_component = min(
            1.0, 0.5 + quality_delta * 3
        )
    else:
        # No correction needed = already good
        if initial_quality >= 0.75:
            improvement_component = 0.85
        else:
            improvement_component = 0.40

    iter_efficiency = 1.0 - (
        (total_iters - 1) / max(max_iters, 1) * 0.2
    )

    sc_score = round(
        improvement_component   * 0.45 +
        error_detection_acc     * 0.35 +
        iter_efficiency         * 0.20,
        4
    )
    sc_score = min(1.0, max(0.0, sc_score))

    return {
        "id"                   : qa_item["id"],
        "question"             : question,
        "reference"            : reference,
        "category"             : category,
        "difficulty"           : difficulty,
        "initial_answer"       : initial_answer,
        "final_answer"         : current_answer,
        "initial_quality"      : round(initial_quality, 4),
        "final_quality"        : round(final_quality,   4),
        "quality_delta"        : quality_delta,
        "pct_improvement"      : pct_improvement,
        "correction_made"      : correction_made,
        "over_corrected"       : over_corrected,
        "total_iterations"     : total_iters,
        "issues_found"         : list(set(all_issues))[:5],
        "quality_flags"        : list(set(all_flags)),
        "real_issues_count"    : n_real,
        "error_detection_acc"  : error_detection_acc,
        "self_correction_score": sc_score,
        "iterations"           : [
            {
                "iter"   : it["iter"],
                "quality": it["quality"],
                "type"   : it["type"],
            }
            for it in iterations
        ],
    }


# ==================================================
# AGGREGATE
# ==================================================

def aggregate_results(results: list) -> dict:

    total     = len(results)
    corrected = [r for r in results if r["correction_made"]]
    improved  = [
        r for r in results if r["quality_delta"] > 0
    ]
    over_corr = [r for r in results if r["over_corrected"]]
    no_issues = [
        r for r in results
        if not r["correction_made"]
        and r["initial_quality"] >= 0.75
    ]

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_stats = {}
    for cat, items in by_cat.items():
        avg_init = float(np.mean([
            i["initial_quality"] for i in items
        ]))
        avg_final = float(np.mean([
            i["final_quality"] for i in items
        ]))
        cat_stats[cat] = {
            "count"           : len(items),
            "avg_sc_score"    : round(float(np.mean([
                i["self_correction_score"]
                for i in items
            ])), 4),
            "correction_rate" : round(sum(
                1 for i in items
                if i["correction_made"]
            ) / len(items), 4),
            "avg_improvement" : round(float(np.mean([
                i["quality_delta"] for i in items
            ])), 4),
            "avg_initial_q"   : round(avg_init,  4),
            "avg_final_q"     : round(avg_final, 4),
            "avg_det_acc"     : round(float(np.mean([
                i["error_detection_acc"] for i in items
            ])), 4),
        }

    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    diff_stats = {}
    for diff, items in by_diff.items():
        diff_stats[diff] = {
            "count"           : len(items),
            "avg_sc_score"    : round(float(np.mean([
                i["self_correction_score"]
                for i in items
            ])), 4),
            "correction_rate" : round(sum(
                1 for i in items
                if i["correction_made"]
            ) / len(items), 4),
            "avg_improvement" : round(float(np.mean([
                i["quality_delta"] for i in items
            ])), 4),
            "avg_det_acc"     : round(float(np.mean([
                i["error_detection_acc"] for i in items
            ])), 4),
        }

    # Issue frequency analysis
    all_flags = []
    for r in results:
        all_flags.extend(r.get("quality_flags", []))

    from collections import Counter
    flag_freq = dict(Counter(all_flags).most_common())

    return {
        "total_evaluated"           : total,
        "timestamp"                 : datetime.now().isoformat(),

        # Core scores
        "avg_self_correction_score" : round(float(np.mean([
            r["self_correction_score"] for r in results
        ])), 4),
        "avg_error_detection_acc"   : round(float(np.mean([
            r["error_detection_acc"] for r in results
        ])), 4),
        "avg_quality_delta"         : round(float(np.mean([
            r["quality_delta"] for r in results
        ])), 4),
        "avg_pct_improvement"       : round(float(np.mean([
            r["pct_improvement"] for r in results
        ])), 2),
        "avg_initial_quality"       : round(float(np.mean([
            r["initial_quality"] for r in results
        ])), 4),
        "avg_final_quality"         : round(float(np.mean([
            r["final_quality"] for r in results
        ])), 4),
        "avg_iterations"            : round(float(np.mean([
            r["total_iterations"] for r in results
        ])), 2),

        # Rates
        "correction_rate"           : round(
            len(corrected) / max(total, 1), 4
        ),
        "improvement_rate"          : round(
            len(improved) / max(total, 1), 4
        ),
        "over_correction_rate"      : round(
            len(over_corr) / max(total, 1), 4
        ),
        "already_good_rate"         : round(
            len(no_issues) / max(total, 1), 4
        ),

        # Counts
        "questions_corrected"       : len(corrected),
        "questions_improved"        : len(improved),
        "questions_over_corrected"  : len(over_corr),
        "questions_already_good"    : len(no_issues),

        # Breakdowns
        "by_category"               : cat_stats,
        "by_difficulty"             : diff_stats,
        "issue_frequency"           : flag_freq,
    }


# ==================================================
# GRAPHS
# ==================================================

def plot_self_correction_overview(
    results : list,
    summary : dict
):

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    # ── 1. Quality Before vs After (Scatter) ─────
    initial = [r["initial_quality"] for r in results]
    final   = [r["final_quality"]   for r in results]

    point_colors = [
        COLORS["good"] if f > i + 0.005 else
        COLORS["bad"]  if f < i - 0.005 else
        COLORS["warn"]
        for i, f in zip(initial, final)
    ]

    axes[0].scatter(
        initial, final,
        c=point_colors, alpha=0.7, s=40, zorder=3
    )
    mn = min(min(initial), min(final)) - 0.02
    mx = max(max(initial), max(final)) + 0.02
    axes[0].plot(
        [mn, mx], [mn, mx],
        color="white", linestyle="--",
        alpha=0.5, label="No change"
    )
    axes[0].set_xlabel("Initial Quality", fontsize=11)
    axes[0].set_ylabel("Final Quality",   fontsize=11)
    axes[0].set_title(
        "Quality: Before vs After Correction",
        fontsize=12, fontweight="bold"
    )
    # Legend patches
    import matplotlib.patches as mpatches
    axes[0].legend(handles=[
        mpatches.Patch(color=COLORS["good"],
                       label="Improved"),
        mpatches.Patch(color=COLORS["bad"],
                       label="Worsened"),
        mpatches.Patch(color=COLORS["warn"],
                       label="No change"),
        mpatches.Patch(color="white",
                       label="No change line",
                       alpha=0.5),
    ], fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── 2. Quality Delta Distribution ────────────
    deltas = [r["quality_delta"] for r in results]
    n_pos  = sum(1 for d in deltas if d > 0.005)
    n_neg  = sum(1 for d in deltas if d < -0.005)
    n_same = len(deltas) - n_pos - n_neg

    axes[1].hist(
        deltas, bins=20,
        color=COLORS["blue"], alpha=0.8,
        edgecolor="white", linewidth=0.5
    )
    axes[1].axvline(
        x=0, color="white",
        linestyle="--", linewidth=2,
        label="No change"
    )
    axes[1].axvline(
        x=float(np.mean(deltas)),
        color=COLORS["gold"], linestyle="-",
        linewidth=2,
        label=f"Mean Δ ({np.mean(deltas):+.4f})"
    )
    axes[1].set_xlabel("Quality Delta (Final − Initial)",
                        fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(
        f"Quality Change Distribution\n"
        f"Improved: {n_pos}  |  "
        f"Same: {n_same}  |  "
        f"Worse: {n_neg}",
        fontsize=11, fontweight="bold"
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # ── 3. SC Score by Category ───────────────────
    cats      = sorted(
        summary["by_category"].keys(),
        key=lambda c: summary["by_category"][c]["avg_sc_score"],
        reverse=True
    )
    sc_scores = [
        summary["by_category"][c]["avg_sc_score"]
        for c in cats
    ]
    det_accs  = [
        summary["by_category"][c]["avg_det_acc"]
        for c in cats
    ]

    x     = np.arange(len(cats))
    width = 0.38

    b1 = axes[2].bar(
        x - width/2, sc_scores,
        width=width, label="SC Score",
        color=COLORS["blue"], alpha=0.85,
        edgecolor="white", linewidth=0.4
    )
    b2 = axes[2].bar(
        x + width/2, det_accs,
        width=width, label="Error Detection Acc",
        color=COLORS["purple"], alpha=0.85,
        edgecolor="white", linewidth=0.4
    )

    for bar, val in zip(b1, sc_scores):
        axes[2].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=7, color="white",
            fontweight="bold"
        )
    for bar, val in zip(b2, det_accs):
        axes[2].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=7, color="#cc99ff",
            fontweight="bold"
        )

    axes[2].axhline(
        y=0.7, color=COLORS["warn"],
        linestyle="--", linewidth=1.2,
        label="Target (0.70)"
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        cats, rotation=30, ha="right", fontsize=8
    )
    axes[2].set_ylim(0, 1.2)
    axes[2].set_title(
        "SC Score & Error Detection by Category",
        fontsize=11, fontweight="bold"
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    # ── 4. Outcomes Summary (Safe Pie) ───────────
    n_total   = summary["total_evaluated"]
    n_better  = summary["questions_improved"]
    n_worse   = summary["questions_over_corrected"]
    n_good    = summary["questions_already_good"]
    n_tried   = summary["questions_corrected"]
    n_failed  = max(0, n_tried - n_better - n_worse)
    n_no_try  = max(0, n_total - n_tried - n_good)

    pie_raw = [
        (n_better, "Improved",       COLORS["good"]),
        (n_worse,  "Over-corrected", COLORS["bad"]),
        (n_good,   "Already Good",   COLORS["blue"]),
        (n_failed, "Tried/No gain",  COLORS["warn"]),
        (n_no_try, "Not corrected",  COLORS["subtext"]),
    ]
    pie_data = [(v,l,c) for v,l,c in pie_raw if v > 0]

    if pie_data:
        pv = [d[0] for d in pie_data]
        pl = [f"{d[1]}\n({d[0]})" for d in pie_data]
        pc = [d[2] for d in pie_data]

        axes[3].pie(
            pv, labels=pl, colors=pc,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={
                "edgecolor":"#0d0d14","linewidth":2
            },
            textprops={
                "color":"white","fontsize":9
            }
        )
    else:
        axes[3].text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            fontsize=12, color=COLORS["subtext"],
            transform=axes[3].transAxes
        )

    axes[3].set_title(
        "Self-Correction Outcomes\n"
        f"(n={n_total} questions)",
        fontsize=12, fontweight="bold"
    )

    # ── 5. Issue Frequency Bar ────────────────────
    freq   = summary.get("issue_frequency", {})
    clean_labels = {
        "length"       : "Too Short",
        "verbose"      : "Too Verbose",
        "medical_terms": "Missing Med Terms",
        "coverage"     : "Low Coverage",
        "reference_sim": "Low Ref Match",
        "relevance"    : "Low Relevance",
        "vague"        : "Vague Answer",
        "unsafe"       : "Unsafe Claim",
    }

    if freq:
        sorted_freq = sorted(
            freq.items(), key=lambda x: x[1],
            reverse=True
        )
        f_labels = [
            clean_labels.get(k, k)
            for k, _ in sorted_freq
        ]
        f_counts = [v for _, v in sorted_freq]
        f_colors = [
            COLORS["bad"]  if c > 10 else
            COLORS["warn"] if c > 5  else
            COLORS["blue"]
            for c in f_counts
        ]

        bars5 = axes[4].barh(
            f_labels, f_counts,
            color=f_colors, alpha=0.85,
            edgecolor="white", linewidth=0.4
        )
        for bar, cnt in zip(bars5, f_counts):
            axes[4].text(
                bar.get_width()+0.1,
                bar.get_y()+bar.get_height()/2,
                str(cnt),
                va="center", fontsize=10,
                fontweight="bold", color="white"
            )
        axes[4].set_xlabel("Occurrences", fontsize=11)
        axes[4].set_title(
            "Most Common Issues Found",
            fontsize=12, fontweight="bold"
        )
        axes[4].grid(axis="x", alpha=0.3)
    else:
        axes[4].text(
            0.5, 0.5, "No issues detected",
            ha="center", va="center",
            fontsize=12, color=COLORS["subtext"],
            transform=axes[4].transAxes
        )
        axes[4].set_title(
            "Issue Frequency",
            fontsize=12, fontweight="bold"
        )

    # ── 6. Quality Before vs After by Difficulty ─
    diffs      = sorted(summary["by_difficulty"].keys())
    init_quals = [
        summary["by_difficulty"][d]["avg_improvement"] + \
        summary["by_difficulty"][d].get(
            "avg_sc_score", 0
        ) * 0.5
        for d in diffs
    ]
    sc_by_diff = [
        summary["by_difficulty"][d]["avg_sc_score"]
        for d in diffs
    ]
    corr_rate  = [
        summary["by_difficulty"][d]["correction_rate"]
        for d in diffs
    ]

    x2     = np.arange(len(diffs))
    width2 = 0.28

    b3 = axes[5].bar(
        x2 - width2, sc_by_diff,
        width=width2, label="SC Score",
        color=COLORS["blue"], alpha=0.85,
        edgecolor="white", linewidth=0.4
    )
    b4 = axes[5].bar(
        x2, corr_rate,
        width=width2, label="Correction Rate",
        color=COLORS["warn"], alpha=0.85,
        edgecolor="white", linewidth=0.4
    )
    b5 = axes[5].bar(
        x2 + width2, [
            summary["by_difficulty"][d]["avg_det_acc"]
            for d in diffs
        ],
        width=width2, label="Error Det Acc",
        color=COLORS["purple"], alpha=0.85,
        edgecolor="white", linewidth=0.4
    )

    for bars_g in [b3, b4, b5]:
        for bar in bars_g:
            val = bar.get_height()
            if val > 0.01:
                axes[5].text(
                    bar.get_x()+bar.get_width()/2,
                    val+0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8, color="white"
                )

    axes[5].set_xticks(x2)
    axes[5].set_xticklabels(
        [d.capitalize() for d in diffs], fontsize=11
    )
    axes[5].set_ylim(0, 1.2)
    axes[5].set_title(
        "SC Metrics by Difficulty",
        fontsize=12, fontweight="bold"
    )
    axes[5].legend(fontsize=9)
    axes[5].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Self-Correction Ability Analysis — NITPY\n"
        f"Evaluated {len(results)} Questions | "
        f"Avg SC Score: "
        f"{summary['avg_self_correction_score']:.4f} | "
        f"Correction Rate: "
        f"{summary['correction_rate']*100:.1f}%",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/self_correction/graphs/"
        "01_self_correction_overview.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_correction_examples(results: list):

    corrected = sorted(
        [r for r in results if r["correction_made"]],
        key=lambda x: x["quality_delta"],
        reverse=True
    )[:5]

    if not corrected:
        # Show all results sorted by quality
        corrected = sorted(
            results,
            key=lambda x: x["initial_quality"]
        )[:3]
        if not corrected:
            print("  ⚠️ No examples to show")
            return

    n_ex   = len(corrected)
    fig, ax = plt.subplots(
        figsize=(18, max(14, n_ex * 4.5 + 3))
    )
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    ax.text(
        0.5, 0.99,
        "Self-Correction Examples",
        ha="center", va="top",
        fontsize=15, fontweight="bold",
        color="white", transform=ax.transAxes
    )
    ax.text(
        0.5, 0.966,
        "Initial answer → Issues detected → Corrected answer",
        ha="center", va="top",
        fontsize=10, color=COLORS["subtext"],
        transform=ax.transAxes, style="italic"
    )

    y     = 0.94
    row_h = 0.86 / n_ex
    gap   = 0.008

    for idx, r in enumerate(corrected):

        y_pos = y - idx * (row_h + gap)

        # Header
        ax.text(
            0.01, y_pos,
            f"#{idx+1}  "
            f"[{r['category'].upper()}]  "
            f"[{r['difficulty'].upper()}]  "
            f"Δ = {r['quality_delta']:+.4f}  "
            f"({r['pct_improvement']:+.1f}%)",
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            color=COLORS["gold"],
            transform=ax.transAxes
        )

        # Question
        q = r["question"][:100]+"..."
        ax.text(
            0.01, y_pos - row_h*0.08,
            f"Q: {q}",
            ha="left", va="top",
            fontsize=8.5, color=COLORS["subtext"],
            transform=ax.transAxes
        )

        # Initial answer
        init = r["initial_answer"][:160]+"..."
        ax.text(
            0.01, y_pos - row_h*0.18,
            f"❌ Initial  "
            f"(quality={r['initial_quality']:.4f})",
            ha="left", va="top",
            fontsize=9, fontweight="bold",
            color=COLORS["bad"],
            transform=ax.transAxes
        )
        ax.text(
            0.02, y_pos - row_h*0.27,
            init,
            ha="left", va="top",
            fontsize=8, color="#ffbbbb",
            transform=ax.transAxes
        )

        # Issues
        issues = r.get("issues_found",[])
        if issues:
            iss_text = " | ".join(issues[:2])[:110]
            ax.text(
                0.01, y_pos - row_h*0.44,
                f"⚠️  Issues: {iss_text}",
                ha="left", va="top",
                fontsize=8.5, color=COLORS["warn"],
                transform=ax.transAxes
            )

        # Final answer
        final = r["final_answer"][:160]+"..."
        ax.text(
            0.01, y_pos - row_h*0.55,
            f"✅ Corrected  "
            f"(quality={r['final_quality']:.4f})",
            ha="left", va="top",
            fontsize=9, fontweight="bold",
            color=COLORS["good"],
            transform=ax.transAxes
        )
        ax.text(
            0.02, y_pos - row_h*0.64,
            final,
            ha="left", va="top",
            fontsize=8, color="#aaffaa",
            transform=ax.transAxes
        )

        # Iters used
        ax.text(
            0.85, y_pos - row_h*0.18,
            f"Iters: {r['total_iterations']}\n"
            f"Det Acc: {r['error_detection_acc']:.2f}",
            ha="left", va="top",
            fontsize=8, color=COLORS["teal"],
            transform=ax.transAxes
        )

        # Divider
        ax.plot(
            [0.01, 0.99],
            [y_pos - row_h*0.88,
             y_pos - row_h*0.88],
            color=COLORS["grid"],
            linewidth=0.8,
            transform=ax.transAxes
        )

    path = (
        "results/self_correction/graphs/"
        "02_correction_examples.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ {path}")


def plot_sc_heatmap(summary: dict):

    cats = sorted(
        summary["by_category"].keys(),
        key=lambda c: summary["by_category"][c]["avg_sc_score"],
        reverse=True
    )
    if not cats:
        print("  ⚠️ No category data — skipping")
        return

    metrics = [
        ("SC Score",      "avg_sc_score"),
        ("Correction %",  "correction_rate"),
        ("Avg Improve",   "avg_improvement"),
        ("Error Det Acc", "avg_det_acc"),
    ]

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(22, 8)
    )

    for idx, (name, key) in enumerate(metrics):
        ax   = axes[idx]
        vals = [
            summary["by_category"][c].get(key, 0)
            for c in cats
        ]

        threshold = 0.6 if "Score" in name else 0.3
        colors_b  = [
            COLORS["good"] if v >= threshold else
            COLORS["warn"] if v >= threshold * 0.7 else
            COLORS["bad"]
            for v in vals
        ]

        bars = ax.barh(
            cats, vals,
            color=colors_b, alpha=0.85,
            edgecolor="white", linewidth=0.4
        )
        for bar, val in zip(bars, vals):
            ax.text(
                max(bar.get_width(), 0) + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}",
                va="center", fontsize=9,
                fontweight="bold", color="white"
            )

        all_vals = [abs(v) for v in vals]
        ax.set_xlim(
            min(min(vals)-0.05, -0.05),
            max(max(vals)+0.12, 0.2)
        )
        ax.set_title(
            name, fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(
            x=0, color="white",
            linewidth=0.8, alpha=0.5
        )

    fig.suptitle(
        "Self-Correction Metrics — Category Breakdown",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = (
        "results/self_correction/graphs/"
        "03_sc_by_category.png"
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

    print(f"\n{'='*65}")
    print(f"  NITPY — SELF-CORRECTION ABILITY REPORT")
    print(f"{'='*65}")
    print(
        f"  Questions Evaluated      : "
        f"{summary['total_evaluated']}"
    )

    print(f"\n  ── Core Metrics {'─'*48}")
    print(
        f"  Self-Correction Score    : "
        f"{summary['avg_self_correction_score']}"
    )
    print(
        f"  Error Detection Accuracy : "
        f"{summary['avg_error_detection_acc']}"
    )
    print(
        f"  Avg Quality Delta        : "
        f"{summary['avg_quality_delta']:+.4f}"
    )
    print(
        f"  Avg % Improvement        : "
        f"{summary['avg_pct_improvement']:+.2f}%"
    )
    print(
        f"  Avg Initial Quality      : "
        f"{summary['avg_initial_quality']}"
    )
    print(
        f"  Avg Final Quality        : "
        f"{summary['avg_final_quality']}"
    )
    print(
        f"  Avg Iterations Used      : "
        f"{summary['avg_iterations']}"
    )

    print(f"\n  ── Correction Stats {'─'*45}")
    print(
        f"  Correction Attempted     : "
        f"{summary['correction_rate']*100:.1f}% "
        f"({summary['questions_corrected']} questions)"
    )
    print(
        f"  Improvement Rate         : "
        f"{summary['improvement_rate']*100:.1f}% "
        f"({summary['questions_improved']} improved)"
    )
    print(
        f"  Over-Correction Rate     : "
        f"{summary['over_correction_rate']*100:.1f}% "
        f"({summary['questions_over_corrected']} worse)"
    )
    print(
        f"  Already Good (no fix)    : "
        f"{summary['already_good_rate']*100:.1f}% "
        f"({summary['questions_already_good']} questions)"
    )

    if summary.get("issue_frequency"):
        print(f"\n  ── Most Common Issues {'─'*43}")
        clean = {
            "length"       : "Answer Too Short",
            "verbose"      : "Answer Too Long",
            "medical_terms": "Missing Medical Terms",
            "coverage"     : "Low Question Coverage",
            "reference_sim": "Low Reference Match",
            "relevance"    : "Low Answer Relevance",
            "vague"        : "Vague/Evasive Answer",
            "unsafe"       : "Unsafe Medical Claim",
        }
        for flag, cnt in sorted(
            summary["issue_frequency"].items(),
            key=lambda x: x[1], reverse=True
        )[:5]:
            label = clean.get(flag, flag)
            print(f"  {label:<30} : {cnt}")

    print(f"\n  ── By Category {'─'*49}")
    print(
        f"  {'Category':<20} "
        f"{'SC Score':>9} "
        f"{'Corr%':>7} "
        f"{'ΔQuality':>9} "
        f"{'DetAcc':>8}"
    )
    print(f"  {'─'*58}")
    for cat, data in sorted(
        summary["by_category"].items(),
        key=lambda x: x[1]["avg_sc_score"],
        reverse=True
    ):
        st = "✅" if data["avg_sc_score"] >= 0.7 else "⚠️"
        print(
            f"  {st} {cat:<18} "
            f"{data['avg_sc_score']:>9.4f} "
            f"{data['correction_rate']*100:>6.1f}% "
            f"{data['avg_improvement']:>+9.4f} "
            f"{data['avg_det_acc']:>8.4f}"
        )

    print(f"\n  ── By Difficulty {'─'*47}")
    for diff, data in sorted(
        summary["by_difficulty"].items(),
        key=lambda x: x[1]["avg_sc_score"],
        reverse=True
    ):
        st = "✅" if data["avg_sc_score"] >= 0.7 else "⚠️"
        print(
            f"  {st} {diff:<12} "
            f"SC: {data['avg_sc_score']:.4f}  "
            f"Correction: {data['correction_rate']*100:.1f}%  "
            f"DetAcc: {data['avg_det_acc']:.4f}  "
            f"n={data['count']}"
        )

    print(f"\n{'='*65}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("\n" + "="*65)
    print("  NITPY — SELF-CORRECTION EVALUATION")
    print("="*65)

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

    print("\nLoading ChromaDB...")
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

    # Use first 50 — change to qa_data for full 200
    eval_data = qa_data[:50]

    print(
        f"\nEvaluating {len(eval_data)} questions "
        f"with self-correction (max 3 iters each)..."
    )
    print("="*65)

    all_results = []

    for i, qa in enumerate(eval_data):
        print(
            f"[{i+1}/{len(eval_data)}] "
            f"{qa['q'][:55]}..."
        )

        try:
            result = evaluate_self_correction(
                qa_item    = qa,
                collection = collection,
                model      = model,
                max_iters  = 3
            )
            all_results.append(result)

            delta   = result["quality_delta"]
            marker  = (
                "✅ improved" if delta > 0.005 else
                "❌ worse"   if delta < -0.005 else
                "→  same"
            )
            n_iss   = result["real_issues_count"]
            det_acc = result["error_detection_acc"]

            print(
                f"  {marker}  "
                f"Init: {result['initial_quality']:.4f}  "
                f"Final: {result['final_quality']:.4f}  "
                f"Δ: {delta:+.4f}  "
                f"Issues: {n_iss}  "
                f"DetAcc: {det_acc:.2f}  "
                f"Iters: {result['total_iterations']}"
            )

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("  ❌ No results.")
        sys.exit(1)

    print(f"\nAggregating {len(all_results)} results...")
    summary = aggregate_results(all_results)

    print("\nGenerating graphs...")
    plot_self_correction_overview(all_results, summary)
    plot_correction_examples(all_results)
    plot_sc_heatmap(summary)

    print_report(summary)

    path = (
        "results/self_correction/"
        "self_correction_report.json"
    )
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Report saved → {path}")
    print("\n  Open graphs:")
    print("  open results/self_correction/graphs/")
    print("="*65)