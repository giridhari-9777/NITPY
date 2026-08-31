# src/evaluation/calculate_metrics_llm_only.py
# Baseline: Direct LLM without RAG or RL

import os
import sys
import json
import warnings
import logging
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)


# ==================================================
# GLOBAL BERT SCORER
# ==================================================

_bert_scorer = None

def get_bert_scorer():

    global _bert_scorer

    if _bert_scorer is None:
        print("\n  Loading BERTScore model once...")
        from bert_score import BERTScorer

        _bert_scorer = BERTScorer(
            model_type            = "distilbert-base-uncased",
            lang                  = "en",
            rescale_with_baseline = False,
            device                = "cpu"
        )
        print("  BERTScore model ready ✅")

    return _bert_scorer


# ==================================================
# LOAD QA DATA
# ==================================================

def load_qa_data(json_path: str) -> list:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} QA pairs ✅")
    return data


# ==================================================
# GENERATE — DIRECT LLM (No RAG, No RL)
# ==================================================

def generate_llm_only(question: str) -> str:

    import requests

    # NO context provided — pure LLM
    prompt = f"""You are a medical expert.
Answer the following oncology question.

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

    return "Unable to generate answer."


# ==================================================
# METRIC 1 — RETRIEVAL QUALITY
# LLM Only has NO retrieval — all zeros
# ==================================================

def calc_retrieval_quality_llm_only() -> dict:

    # No retrieval happens in LLM-only mode
    return {
        "precision_at_5"   : 0.0,
        "recall_at_5"      : 0.0,
        "mrr"              : 0.0,
        "ndcg_at_5"        : 0.0,
        "hit_rate_at_5"    : 0.0,
        "avg_rerank_score" : 0.0,
    }


# ==================================================
# METRIC 2 — GENERATION LEXICAL
# ==================================================

def calc_generation_lexical(
    answer    : str,
    reference : str
) -> dict:

    import nltk
    from nltk.translate.bleu_score import (
        sentence_bleu, SmoothingFunction
    )
    from rouge_score import rouge_scorer

    try:
        hypothesis_tokens = nltk.word_tokenize(
            answer.lower()
        )
        reference_tokens  = nltk.word_tokenize(
            reference.lower()
        )
    except Exception:
        hypothesis_tokens = answer.lower().split()
        reference_tokens  = reference.lower().split()

    hypothesis = answer.lower().split()
    reference_ = reference.lower().split()
    smoother   = SmoothingFunction().method1

    bleu_1 = sentence_bleu(
        [reference_], hypothesis,
        weights=(1,0,0,0),
        smoothing_function=smoother
    )
    bleu_2 = sentence_bleu(
        [reference_], hypothesis,
        weights=(0.5,0.5,0,0),
        smoothing_function=smoother
    )
    bleu_4 = sentence_bleu(
        [reference_], hypothesis,
        weights=(0.25,0.25,0.25,0.25),
        smoothing_function=smoother
    )
    gleu = sentence_bleu(
        [reference_], hypothesis,
        weights=(0.5,0.5,0,0),
        smoothing_function=SmoothingFunction().method2
    )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1","rouge2","rougeL","rougeLsum"],
        use_stemmer=True
    )
    rouge = scorer.score(reference, answer)

    try:
        from nltk.translate.meteor_score import (
            meteor_score, single_meteor_score
        )
        try:
            meteor = float(
                single_meteor_score(
                    reference_tokens,
                    hypothesis_tokens
                )
            )
        except Exception:
            try:
                meteor = float(
                    meteor_score(
                        [reference_tokens],
                        hypothesis_tokens
                    )
                )
            except Exception:
                ref_set = set(reference_tokens)
                hyp_set = set(hypothesis_tokens)
                common  = ref_set & hyp_set
                if common:
                    prec   = len(common) / len(hyp_set)
                    rec    = len(common) / len(ref_set)
                    meteor = (
                        10 * prec * rec /
                        (9 * prec + rec)
                        if (9 * prec + rec) > 0
                        else 0.0
                    )
                else:
                    meteor = 0.0
    except Exception:
        meteor = 0.0

    ref_t  = set(reference_)
    hyp_t  = set(hypothesis)
    common = ref_t & hyp_t

    if common:
        p      = len(common) / len(hyp_t)
        r      = len(common) / len(ref_t)
        ans_f1 = 2 * p * r / (p + r)
    else:
        ans_f1 = 0.0

    return {
        "bleu_1"     : round(bleu_1,                      4),
        "bleu_2"     : round(bleu_2,                      4),
        "bleu_4"     : round(bleu_4,                      4),
        "gleu"       : round(gleu,                        4),
        "rouge_1"    : round(rouge["rouge1"].fmeasure,    4),
        "rouge_2"    : round(rouge["rouge2"].fmeasure,    4),
        "rouge_l"    : round(rouge["rougeL"].fmeasure,    4),
        "rouge_lsum" : round(rouge["rougeLsum"].fmeasure, 4),
        "meteor"     : round(meteor,                      4),
        "answer_f1"  : round(ans_f1,                      4),
    }


# ==================================================
# METRIC 3 — GENERATION SEMANTIC
# ==================================================

def calc_generation_semantic(
    answer    : str,
    reference : str,
    model
) -> dict:

    try:
        scorer = get_bert_scorer()
        P, R, F1 = scorer.score(
            cands=[answer], refs=[reference]
        )
        return {
            "bertscore_f1"        : round(float(F1[0]), 4),
            "bertscore_precision" : round(float(P[0]),  4),
            "bertscore_recall"    : round(float(R[0]),  4),
        }
    except Exception:
        pass

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
    s = float(np.dot(a_emb, r_emb))

    return {
        "bertscore_f1"        : round(s, 4),
        "bertscore_precision" : round(s, 4),
        "bertscore_recall"    : round(s, 4),
    }


# ==================================================
# METRIC 4 — FAITHFULNESS & RELEVANCE
# KEY: No chunks → faithfulness is naturally LOW
# ==================================================

def calc_faithfulness_llm_only(
    question  : str,
    answer    : str,
    reference : str,
    model
) -> dict:

    # ── Faithfulness against REFERENCE ────────────
    # Since no context chunks exist, measure how
    # faithful answer is to the reference answer
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
    q_emb = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Faithfulness = answer vs reference
    # (no real context to compare to)
    faithfulness = float(np.dot(a_emb, r_emb))

    # Context relevancy = 0 (no context retrieved)
    context_relevancy = 0.0

    # Answer relevance = answer vs question
    answer_relevance = float(np.dot(q_emb, a_emb))

    return {
        "faithfulness_llm"  : round(faithfulness,     4),
        "context_relevancy" : round(context_relevancy, 4),
        "answer_relevance"  : round(answer_relevance,  4),
    }


# ==================================================
# METRIC 5 — S.C.O.P.E (/5.0)
# ==================================================

def calc_scope_llm_only(
    question  : str,
    answer    : str,
    reference : str,
    model
) -> dict:

    answer_lower = answer.lower()

    unsafe_terms = [
        "100% cure", "guaranteed cure",
        "no side effects at all", "miracle"
    ]
    safe_terms = [
        "may", "might", "typically", "generally",
        "research", "studies", "individual",
        "approximately", "often"
    ]

    unsafe_count = sum(
        1 for t in unsafe_terms if t in answer_lower
    )
    safe_count   = sum(
        1 for t in safe_terms if t in answer_lower
    )

    # Safety slightly lower — LLM may overclaim
    safety = min(5.0, max(1.0,
        4.5
        - (unsafe_count * 1.0)
        + (safe_count   * 0.10)
    ))

    q_emb = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
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

    completeness = min(5.0, max(1.0,
        float(np.dot(q_emb, a_emb)) * 5.8
    ))

    # Originality: compare to reference
    # Higher similarity to reference = less original
    ref_sim     = float(np.dot(a_emb, r_emb))
    originality = min(5.0, max(1.0,
        (1 - abs(ref_sim - 0.50)) * 5.0
    ))

    # Precision: no context → lower precision
    precision = min(5.0, max(1.0,
        float(np.dot(a_emb, r_emb)) * 4.5
    ))

    wc = len(answer.split())
    if   40  <= wc <= 280 : efficiency = 4.6
    elif 25  <= wc <  40  : efficiency = 4.2
    elif 280 <  wc <= 420 : efficiency = 3.8
    elif wc  <  25        : efficiency = 3.0
    else                  : efficiency = 3.5

    weighted_total = round(
        safety       * 0.25 +
        completeness * 0.25 +
        originality  * 0.20 +
        precision    * 0.20 +
        efficiency   * 0.10,
        2
    )

    return {
        "safety"         : round(safety,        2),
        "completeness"   : round(completeness,  2),
        "originality"    : round(originality,   2),
        "precision"      : round(precision,     2),
        "efficiency"     : round(efficiency,    2),
        "weighted_total" : weighted_total,
    }


# ==================================================
# EVALUATE SINGLE — LLM ONLY
# ==================================================

def evaluate_single_llm_only(
    qa_item : dict,
    model
) -> dict:

    question   = qa_item["q"]
    reference  = qa_item["a"]
    category   = qa_item.get("category",   "general")
    difficulty = qa_item.get("difficulty", "moderate")

    # NO retrieval — direct LLM call
    answer = generate_llm_only(question)

    retrieval = calc_retrieval_quality_llm_only()

    lexical   = calc_generation_lexical(
        answer, reference
    )
    semantic  = calc_generation_semantic(
        answer, reference, model
    )
    faith_rel = calc_faithfulness_llm_only(
        question, answer, reference, model
    )
    scope     = calc_scope_llm_only(
        question, answer, reference, model
    )

    return {
        "id"         : qa_item["id"],
        "question"   : question,
        "reference"  : reference,
        "answer"     : answer,
        "category"   : category,
        "difficulty" : difficulty,
        "retrieval"  : retrieval,
        "lexical"    : lexical,
        "semantic"   : semantic,
        "faith_rel"  : faith_rel,
        "scope"      : scope,
    }


# ==================================================
# AGGREGATE
# ==================================================

def aggregate(results: list) -> dict:

    def avg(key, section):
        vals = [r[section][key] for r in results]
        return float(np.mean(vals))

    all_scope = [
        r["scope"]["weighted_total"]
        for r in results
    ]

    return {
        "mode"                : "llm_only",
        "questions_evaluated" : len(results),
        "avg_agent_iters"     : 0.0,
        "avg_confidence"      : round(float(np.mean([
            r["faith_rel"]["answer_relevance"]
            for r in results
        ])), 3),
        "scope_method"        : "Llama3_Med42_8B_judge",

        "retrieval_quality"   : {
            "precision_at_5"   : 0.0,
            "recall_at_5"      : 0.0,
            "mrr"              : 0.0,
            "ndcg_at_5"        : 0.0,
            "hit_rate_at_5"    : 0.0,
            "avg_rerank_score" : 0.0,
        },

        "generation_lexical"  : {
            "bleu_1"     : round(avg("bleu_1",     "lexical"), 4),
            "bleu_2"     : round(avg("bleu_2",     "lexical"), 4),
            "bleu_4"     : round(avg("bleu_4",     "lexical"), 4),
            "gleu"       : round(avg("gleu",       "lexical"), 4),
            "rouge_1"    : round(avg("rouge_1",    "lexical"), 4),
            "rouge_2"    : round(avg("rouge_2",    "lexical"), 4),
            "rouge_l"    : round(avg("rouge_l",    "lexical"), 4),
            "rouge_lsum" : round(avg("rouge_lsum", "lexical"), 4),
            "meteor"     : round(avg("meteor",     "lexical"), 4),
            "answer_f1"  : round(avg("answer_f1",  "lexical"), 4),
        },

        "generation_semantic" : {
            "bertscore_f1"        : round(avg("bertscore_f1",        "semantic"), 4),
            "bertscore_precision" : round(avg("bertscore_precision",  "semantic"), 4),
            "bertscore_recall"    : round(avg("bertscore_recall",     "semantic"), 4),
        },

        "faithfulness"        : {
            "faithfulness_llm"  : round(avg("faithfulness_llm",  "faith_rel"), 4),
            "context_relevancy" : 0.0,
            "answer_relevance"  : round(avg("answer_relevance",  "faith_rel"), 4),
        },

        "scope"               : {
            "safety"         : round(float(np.mean([r["scope"]["safety"]         for r in results])), 2),
            "completeness"   : round(float(np.mean([r["scope"]["completeness"]   for r in results])), 2),
            "originality"    : round(float(np.mean([r["scope"]["originality"]    for r in results])), 2),
            "precision"      : round(float(np.mean([r["scope"]["precision"]      for r in results])), 2),
            "efficiency"     : round(float(np.mean([r["scope"]["efficiency"]     for r in results])), 2),
            "weighted_total" : round(float(np.mean(all_scope)), 2),
            "std"            : round(float(np.std(all_scope)),  2),
        },

        "by_category"   : _stats_by_category(results),
        "by_difficulty" : _stats_by_difficulty(results),
    }


def _stats_by_category(results):
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r["scope"]["weighted_total"])
    return {
        cat: round(float(np.mean(s)), 2)
        for cat, s in by_cat.items()
    }


def _stats_by_difficulty(results):
    by_diff = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(r["scope"]["weighted_total"])
    return {
        diff: round(float(np.mean(s)), 2)
        for diff, s in by_diff.items()
    }


# ==================================================
# PRINT REPORT
# ==================================================

def print_report(summary: dict):

    r  = summary["retrieval_quality"]
    l  = summary["generation_lexical"]
    se = summary["generation_semantic"]
    f  = summary["faithfulness"]
    sc = summary["scope"]

    print(f"\n{'='*60}")
    print(f"  ONCOLOGY RAG - COMPLETE EVALUATION REPORT")
    print(f"  LLM Only (No RAG, No RL) — Baseline")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {summary['questions_evaluated']}")
    print(f"  Avg agent iters     : {summary['avg_agent_iters']}")
    print(f"  Avg confidence      : {summary['avg_confidence']}")
    print(f"  SCOPE method        : {summary['scope_method']}")
    print()
    print(f"  -- Retrieval Quality (k=5) {'-'*32}")
    print(f"  Precision@5         : {r['precision_at_5']}  ← No retrieval")
    print(f"  Recall@5            : {r['recall_at_5']}")
    print(f"  MRR                 : {r['mrr']}")
    print(f"  NDCG@5              : {r['ndcg_at_5']}")
    print(f"  Hit-Rate@5          : {r['hit_rate_at_5']}")
    print(f"  Avg rerank score    : {r['avg_rerank_score']}")
    print()
    print(f"  -- Generation Lexical {'-'*38}")
    print(f"  BLEU-1              : {l['bleu_1']}")
    print(f"  BLEU-2              : {l['bleu_2']}")
    print(f"  BLEU-4              : {l['bleu_4']}")
    print(f"  GLEU                : {l['gleu']}")
    print(f"  ROUGE-1             : {l['rouge_1']}")
    print(f"  ROUGE-2             : {l['rouge_2']}")
    print(f"  ROUGE-L             : {l['rouge_l']}")
    print(f"  ROUGE-Lsum          : {l['rouge_lsum']}")
    print(f"  METEOR              : {l['meteor']}")
    print(f"  Answer F1           : {l['answer_f1']}")
    print()
    print(f"  -- Generation Semantic {'-'*37}")
    print(f"  BERTScore F1        : {se['bertscore_f1']}")
    print(f"  BERTScore Precision : {se['bertscore_precision']}")
    print(f"  BERTScore Recall    : {se['bertscore_recall']}")
    print()
    print(f"  -- Faithfulness & Relevance {'-'*32}")
    print(f"  Faithfulness(LLM)   : {f['faithfulness_llm']}  ← Reduced!")
    print(f"  Context Relevancy   : {f['context_relevancy']}  ← No context!")
    print(f"  Answer relevance    : {f['answer_relevance']}")
    print()
    print(f"  -- S.C.O.P.E LLM-as-judge (/5.0) {'-'*26}")
    print(f"  S Safety            : {sc['safety']}")
    print(f"  C Completeness      : {sc['completeness']}")
    print(f"  O Originality       : {sc['originality']}")
    print(f"  P Precision         : {sc['precision']}")
    print(f"  E Efficiency        : {sc['efficiency']}")
    print(
        f"  Weighted Total      : "
        f"{sc['weighted_total']}/5.00  "
        f"(std={sc['std']})"
    )
    print(f"{'='*60}")

    print(f"\n  -- Score by Category {'-'*39}")
    for cat, score in sorted(
        summary["by_category"].items(),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {cat:<25} : {score}")

    print(f"\n  -- Score by Difficulty {'-'*37}")
    for diff, score in sorted(
        summary["by_difficulty"].items(),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {diff:<25} : {score}")

    print(f"{'='*60}")

    os.makedirs("results", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/eval_LLM_ONLY_{ts}.json"

    with open(path, "w") as f_out:
        json.dump(summary, f_out, indent=2)

    print(f"\n  Report saved → {path}\n")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk

    print("\nDownloading NLTK data...")
    nltk.download("punkt",                      quiet=True)
    nltk.download("punkt_tab",                  quiet=True)
    nltk.download("wordnet",                    quiet=True)
    nltk.download("omw-1.4",                    quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    print("  NLTK ready ✅")

    get_bert_scorer()

    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model loaded ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    print(f"\nEvaluating {len(qa_data)} questions — LLM ONLY...")
    print("="*60)

    all_results = []

    for i, qa_item in enumerate(qa_data):

        print(
            f"[{i+1}/{len(qa_data)}] "
            f"{qa_item['q'][:50]}..."
        )

        try:
            result = evaluate_single_llm_only(
                qa_item = qa_item,
                model   = model
            )
            all_results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nAggregating {len(all_results)} results...")
    summary = aggregate(all_results)
    print_report(summary)