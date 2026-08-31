# src/evaluation/calculate_metrics_extended.py
# Extended Agent Evaluation Metrics
# Evaluates HOW the agent works, not just output quality

import os
import sys
import json
import warnings
import logging
import numpy as np
import requests
import re
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
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(
            model_type            = "distilbert-base-uncased",
            lang                  = "en",
            rescale_with_baseline = False,
            device                = "cpu"
        )
    return _bert_scorer


# ==================================================
# LOAD DATA
# ==================================================

def load_qa_data(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} QA pairs ✅")
    return data


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
            "documents","metadatas","distances"
        ]
    )

    try:
        q_tokens = nltk.word_tokenize(question.lower())
    except Exception:
        q_tokens = question.lower().split()

    stopwords = {
        "the","a","an","is","are","was","were",
        "in","on","at","to","of","and","or",
        "but","for","with","by","what","how",
        "why","when","which","who","do","does"
    }
    keywords = [
        t for t in q_tokens
        if t not in stopwords and len(t) > 2
    ]

    chunks = []
    for i in range(len(result["ids"][0])):

        raw_score = 1 - result["distances"][0][i]
        text      = result["documents"][0][i]
        source    = result["metadatas"][0][i].get(
            "source","unknown"
        )

        c_emb = model.encode(
            text[:500],
            normalize_embeddings=True,
            convert_to_numpy=True
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


def generate_answer(question: str, context: str) -> str:

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
# ══════════════════════════════════════════════════
# NEW EXTENDED METRICS
# ══════════════════════════════════════════════════
# ==================================================


# ==================================================
# METRIC A — AGENT EFFICIENCY SCORE
# How efficiently does the agent use iterations?
# ==================================================

def calc_agent_efficiency(
    question    : str,
    answer      : str,
    chunks      : list,
    agent_iters : int,
    model
) -> dict:

    # 1. Did it retrieve relevant chunks?
    q_emb = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    a_emb = model.encode(
        answer[:400],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    chunk_scores = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        chunk_scores.append(float(np.dot(q_emb, c_emb)))

    avg_chunk_relevance = (
        float(np.mean(chunk_scores))
        if chunk_scores else 0.0
    )

    # 2. Answer quality per iteration
    answer_quality = float(np.dot(q_emb, a_emb))

    # 3. Efficiency = quality achieved per iteration
    # Fewer iterations with good quality = more efficient
    iters_used = max(1, agent_iters)
    efficiency = min(1.0, answer_quality / iters_used * 3)

    # 4. Chunk utilization — did agent use top chunks?
    used_chunks = min(len(chunks), 5)
    chunk_util  = used_chunks / 5.0

    # 5. Response length efficiency
    wc = len(answer.split())
    if 30 <= wc <= 200:
        length_eff = 1.0
    elif wc < 30:
        length_eff = wc / 30.0
    else:
        length_eff = 200.0 / wc

    agent_efficiency = round(
        efficiency           * 0.30 +
        avg_chunk_relevance  * 0.30 +
        chunk_util           * 0.20 +
        length_eff           * 0.20,
        4
    )

    return {
        "agent_efficiency_score" : agent_efficiency,
        "answer_quality"         : round(answer_quality,        4),
        "avg_chunk_relevance"    : round(avg_chunk_relevance,   4),
        "chunk_utilization"      : round(chunk_util,            4),
        "length_efficiency"      : round(length_eff,            4),
        "iters_used"             : iters_used,
    }


# ==================================================
# METRIC B — QUERY RESOLUTION RATE
# Did the agent fully resolve the question?
# ==================================================

def calc_query_resolution(
    question : str,
    answer   : str,
    model
) -> dict:

    q_lower = question.lower()
    a_lower = answer.lower()

    # 1. Question type detection
    qtype_keywords = {
        "what"     : ["what","which","name"],
        "how"      : ["how","mechanism","process"],
        "why"      : ["why","reason","cause"],
        "when"     : ["when","stage","timing"],
        "treatment": ["treat","therapy","drug","regimen"],
        "survival" : ["survive","prognosis","rate","percent"],
        "symptoms" : ["symptom","sign","present","manifest"],
        "diagnosis": ["diagnose","detect","test","biopsy"],
    }

    detected_types = []
    for qtype, kws in qtype_keywords.items():
        if any(kw in q_lower for kw in kws):
            detected_types.append(qtype)

    # 2. Answer completeness check
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

    semantic_match = float(np.dot(q_emb, a_emb))

    # 3. Key term coverage
    q_terms = set(question.lower().split())
    a_terms = set(answer.lower().split())

    # Remove stop words
    stops = {
        "the","a","an","is","are","was","were",
        "in","on","at","to","of","and","or","it"
    }
    q_terms = q_terms - stops
    a_terms = a_terms - stops

    term_coverage = (
        len(q_terms & a_terms) / len(q_terms)
        if q_terms else 0.0
    )

    # 4. Resolution indicators
    resolution_phrases = [
        "treatment", "therapy", "diagnosed", "stage",
        "survival", "symptoms", "prognosis", "cells",
        "cancer", "tumor", "chemotherapy", "radiation"
    ]
    resolution_count = sum(
        1 for p in resolution_phrases if p in a_lower
    )
    resolution_density = min(
        1.0, resolution_count / 5.0
    )

    # 5. Evasion check (did agent avoid answering?)
    evasion_phrases = [
        "i don't know", "i cannot", "i'm not sure",
        "please consult", "no information",
        "not available", "unable to answer"
    ]
    has_evasion = any(
        p in a_lower for p in evasion_phrases
    )
    evasion_penalty = 0.3 if has_evasion else 0.0

    query_resolution = round(
        max(0.0,
            semantic_match   * 0.35 +
            term_coverage    * 0.25 +
            resolution_density * 0.25 +
            (0.15 if not has_evasion else 0.0)
            - evasion_penalty
        ),
        4
    )

    return {
        "query_resolution_rate"  : query_resolution,
        "semantic_match"         : round(semantic_match,       4),
        "term_coverage"          : round(term_coverage,        4),
        "resolution_density"     : round(resolution_density,   4),
        "has_evasion"            : has_evasion,
        "detected_question_types": detected_types,
    }


# ==================================================
# METRIC C — CONTEXT UTILIZATION SCORE
# How well does the agent use retrieved context?
# ==================================================

def calc_context_utilization(
    answer : str,
    chunks : list,
    model
) -> dict:

    if not chunks or not answer:
        return {
            "context_utilization_score": 0.0,
            "coverage_depth"           : 0.0,
            "source_diversity"         : 0.0,
            "context_integration"      : 0.0,
            "redundancy_penalty"       : 0.0,
        }

    a_emb = model.encode(
        answer[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 1. Coverage depth — how many chunks contributed?
    chunk_similarities = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        sim = float(np.dot(a_emb, c_emb))
        chunk_similarities.append(sim)

    # Chunks above threshold = contributing chunks
    threshold        = 0.40
    contributing     = [
        s for s in chunk_similarities if s >= threshold
    ]
    coverage_depth   = (
        len(contributing) / len(chunk_similarities)
        if chunk_similarities else 0.0
    )

    # 2. Source diversity — different sources used
    sources = list(set([
        c.get("source","unknown")
        for c in chunks[:5]
    ]))
    source_diversity = min(1.0, len(sources) / 3.0)

    # 3. Context integration — does answer blend chunks?
    if chunk_similarities:
        max_sim  = max(chunk_similarities)
        mean_sim = float(np.mean(chunk_similarities))
        # Good integration = high mean (not just top-1)
        integration = (max_sim * 0.4 + mean_sim * 0.6)
    else:
        integration = 0.0

    # 4. Redundancy penalty
    # If all chunks are too similar = redundant retrieval
    if len(chunk_similarities) >= 2:
        sim_variance = float(np.std(chunk_similarities))
        # Low variance = redundant
        redundancy_penalty = max(0.0, 0.1 - sim_variance)
    else:
        redundancy_penalty = 0.0

    ctx_util = round(
        max(0.0,
            coverage_depth   * 0.30 +
            source_diversity * 0.20 +
            integration      * 0.40 +
            (0.10 if not redundancy_penalty else 0.0)
            - redundancy_penalty
        ),
        4
    )

    return {
        "context_utilization_score": ctx_util,
        "coverage_depth"           : round(coverage_depth,      4),
        "source_diversity"         : round(source_diversity,     4),
        "context_integration"      : round(integration,          4),
        "redundancy_penalty"       : round(redundancy_penalty,   4),
        "contributing_chunks"      : len(contributing),
    }


# ==================================================
# METRIC D — RESPONSE CONSISTENCY SCORE
# Is the answer consistent with the question context?
# ==================================================

def calc_response_consistency(
    question  : str,
    answer    : str,
    reference : str,
    model
) -> dict:

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

    # 1. Q-A semantic consistency
    qa_consistency = float(np.dot(q_emb, a_emb))

    # 2. A-Reference consistency
    ar_consistency = float(np.dot(a_emb, r_emb))

    # 3. Lexical consistency — key medical terms
    medical_terms = [
        "cancer","tumor","stage","treatment","therapy",
        "diagnosis","prognosis","chemotherapy","surgery",
        "radiation","biopsy","metastasis","malignant",
        "benign","oncology","lymph","carcinoma","sarcoma"
    ]

    ref_terms = set([
        t for t in medical_terms
        if t in reference.lower()
    ])
    ans_terms = set([
        t for t in medical_terms
        if t in answer.lower()
    ])

    if ref_terms:
        term_consistency = (
            len(ref_terms & ans_terms) / len(ref_terms)
        )
    else:
        term_consistency = 1.0

    # 4. Contradiction check
    # Simple check: opposite sentiment markers
    positive_q = any(
        w in question.lower()
        for w in [
            "benefit","effective","successful",
            "improve","treat","cure","help"
        ]
    )
    negative_a = any(
        w in answer.lower()
        for w in ["not effective","cannot treat",
                  "no treatment","incurable"]
    )
    contradiction_penalty = (
        0.2 if (positive_q and negative_a) else 0.0
    )

    # 5. Specificity — does answer address specific question?
    q_entities = [
        w for w in question.lower().split()
        if len(w) > 5 and w not in [
            "cancer","tumor","about","which","where"
        ]
    ]
    a_lower = answer.lower()
    entity_hits = sum(
        1 for e in q_entities if e in a_lower
    )
    specificity = (
        min(1.0, entity_hits / max(len(q_entities), 1))
    )

    consistency_score = round(
        max(0.0,
            qa_consistency    * 0.25 +
            ar_consistency    * 0.30 +
            term_consistency  * 0.20 +
            specificity       * 0.25
            - contradiction_penalty
        ),
        4
    )

    return {
        "response_consistency_score": consistency_score,
        "qa_consistency"            : round(qa_consistency,   4),
        "ar_consistency"            : round(ar_consistency,   4),
        "term_consistency"          : round(term_consistency, 4),
        "specificity"               : round(specificity,      4),
        "contradiction_detected"    : contradiction_penalty > 0,
    }


# ==================================================
# METRIC E — EMPATHY SCORE
# Does the agent respond with appropriate empathy?
# ==================================================

def calc_empathy_score(answer: str) -> dict:

    a_lower = answer.lower()

    # Empathy phrase categories
    reassurance = [
        "don't worry", "you are not alone",
        "we can", "together", "i understand",
        "it's natural", "it's okay", "many patients",
        "completely understandable", "here for you",
        "support", "help you", "i'm here"
    ]

    acknowledgment = [
        "that's a", "you're right", "great question",
        "important question", "you're asking",
        "i hear you", "i understand your concern",
        "your concern", "feeling", "scared", "worried"
    ]

    hope = [
        "promising", "effective treatment",
        "good prognosis", "can be treated",
        "options available", "advances in",
        "research shows", "many patients",
        "successful outcomes", "hope"
    ]

    hedging = [
        "may", "might", "typically", "generally",
        "often", "in some cases", "approximately",
        "research suggests", "studies show",
        "consult your", "doctor"
    ]

    # Count matches
    r_count = sum(
        1 for p in reassurance if p in a_lower
    )
    a_count = sum(
        1 for p in acknowledgment if p in a_lower
    )
    h_count = sum(
        1 for p in hope if p in a_lower
    )
    hg_count = sum(
        1 for p in hedging if p in a_lower
    )

    # Score each component
    r_score  = min(1.0, r_count  / 2.0)
    a_score  = min(1.0, a_count  / 2.0)
    h_score  = min(1.0, h_count  / 2.0)
    hg_score = min(1.0, hg_count / 3.0)

    # Tone analysis — avoid harsh words
    harsh_words = [
        "fatal", "deadly", "die", "death",
        "hopeless", "impossible", "never",
        "always fatal", "no cure"
    ]
    harsh_count  = sum(
        1 for w in harsh_words if w in a_lower
    )
    harsh_penalty = min(0.3, harsh_count * 0.1)

    empathy_score = round(
        max(0.0,
            r_score  * 0.35 +
            a_score  * 0.25 +
            h_score  * 0.20 +
            hg_score * 0.20
            - harsh_penalty
        ),
        4
    )

    return {
        "empathy_score"         : empathy_score,
        "reassurance_score"     : round(r_score,      4),
        "acknowledgment_score"  : round(a_score,      4),
        "hope_score"            : round(h_score,      4),
        "hedging_score"         : round(hg_score,     4),
        "harsh_penalty"         : round(harsh_penalty,4),
        "empathy_level"         : (
            "HIGH"   if empathy_score >= 0.6 else
            "MEDIUM" if empathy_score >= 0.3 else
            "LOW"
        ),
    }


# ==================================================
# METRIC F — MEDICAL SAFETY COMPLIANCE
# Does the answer follow medical safety guidelines?
# ==================================================

def calc_medical_safety(
    question : str,
    answer   : str
) -> dict:

    a_lower = answer.lower()
    q_lower = question.lower()

    # 1. Dangerous claim check
    dangerous_claims = [
        "100% cure", "guaranteed cure",
        "definitely will cure", "no side effects at all",
        "miracle cure", "certain death",
        "always fatal", "will definitely die",
        "stop your medication", "don't take",
        "ignore your doctor"
    ]
    dangerous_count = sum(
        1 for c in dangerous_claims if c in a_lower
    )
    safety_score_raw = max(
        0.0, 1.0 - (dangerous_count * 0.4)
    )

    # 2. Appropriate disclaimer check
    disclaimer_phrases = [
        "consult", "doctor", "oncologist",
        "medical professional", "healthcare",
        "physician", "specialist",
        "may vary", "individual", "depending on"
    ]
    has_disclaimer = any(
        p in a_lower for p in disclaimer_phrases
    )

    # 3. Dosage safety
    # Should not give specific drug dosages
    dosage_patterns = [
        r"\d+\s*mg",
        r"\d+\s*ml",
        r"\d+\s*mcg",
        r"take \d+",
        r"dose of \d+"
    ]
    has_specific_dosage = any(
        re.search(pat, a_lower)
        for pat in dosage_patterns
    )
    dosage_penalty = 0.1 if has_specific_dosage else 0.0

    # 4. Evidence-based language
    evidence_phrases = [
        "studies show", "research indicates",
        "clinical trials", "evidence suggests",
        "guidelines recommend", "standard of care",
        "according to", "typically"
    ]
    has_evidence = any(
        p in a_lower for p in evidence_phrases
    )

    # 5. Appropriate scope check
    # Is it answering within oncology scope?
    oncology_terms = [
        "cancer","tumor","oncology","chemotherapy",
        "radiation","surgery","biopsy","staging",
        "metastasis","prognosis","therapy","treatment"
    ]
    in_scope = any(
        t in a_lower for t in oncology_terms
    )
    scope_score = 1.0 if in_scope else 0.5

    medical_safety = round(
        max(0.0,
            safety_score_raw * 0.35 +
            (0.20 if has_disclaimer else 0.05) +
            (0.20 if has_evidence   else 0.05) +
            scope_score              * 0.20 +
            (0.05 if not has_specific_dosage else 0.0)
            - dosage_penalty
        ),
        4
    )

    return {
        "medical_safety_score"    : medical_safety,
        "dangerous_claim_count"   : dangerous_count,
        "has_disclaimer"          : has_disclaimer,
        "has_evidence_language"   : has_evidence,
        "has_specific_dosage"     : has_specific_dosage,
        "in_oncology_scope"       : in_scope,
        "safety_level"            : (
            "SAFE"     if medical_safety >= 0.75 else
            "MODERATE" if medical_safety >= 0.50 else
            "UNSAFE"
        ),
    }


# ==================================================
# METRIC G — KNOWLEDGE COVERAGE SCORE
# How much of the medical knowledge does it cover?
# ==================================================

def calc_knowledge_coverage(
    question  : str,
    answer    : str,
    reference : str,
    model
) -> dict:

    # 1. Reference coverage — how much of reference
    #    answer is covered by generated answer?
    ref_sentences = [
        s.strip() for s in reference.split(".")
        if len(s.strip()) > 10
    ]
    ans_sentences = [
        s.strip() for s in answer.split(".")
        if len(s.strip()) > 10
    ]

    if not ref_sentences:
        return {
            "knowledge_coverage_score": 0.0,
            "reference_coverage"      : 0.0,
            "concept_coverage"        : 0.0,
            "depth_score"             : 0.0,
            "breadth_score"           : 0.0,
        }

    # Semantic coverage of each reference sentence
    covered_count = 0
    for ref_sent in ref_sentences:
        r_emb = model.encode(
            ref_sent,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        best_match = 0.0
        for ans_sent in ans_sentences:
            a_emb = model.encode(
                ans_sent,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            sim = float(np.dot(r_emb, a_emb))
            best_match = max(best_match, sim)

        if best_match >= 0.55:
            covered_count += 1

    reference_coverage = (
        covered_count / len(ref_sentences)
    )

    # 2. Medical concept coverage
    cancer_concepts = {
        "epidemiology": [
            "incidence","prevalence","risk","factor"
        ],
        "pathology"   : [
            "cell","tissue","grade","differentiation"
        ],
        "symptoms"    : [
            "symptom","sign","present","manifest"
        ],
        "diagnosis"   : [
            "diagnose","test","biopsy","scan","mri","ct"
        ],
        "staging"     : [
            "stage","tnm","spread","lymph","node"
        ],
        "treatment"   : [
            "treat","therapy","chemo","radiation","surgery"
        ],
        "prognosis"   : [
            "survive","outcome","prognosis","remission"
        ],
    }

    a_lower   = answer.lower()
    q_lower   = question.lower()
    ref_lower = reference.lower()

    covered_concepts = 0
    total_concepts   = 0

    for concept, keywords in cancer_concepts.items():
        in_reference = any(
            kw in ref_lower for kw in keywords
        )
        if in_reference:
            total_concepts += 1
            in_answer = any(
                kw in a_lower for kw in keywords
            )
            if in_answer:
                covered_concepts += 1

    concept_coverage = (
        covered_concepts / max(total_concepts, 1)
    )

    # 3. Depth score — detail level
    wc = len(answer.split())
    if   wc >= 100 : depth = 1.0
    elif wc >= 60  : depth = 0.8
    elif wc >= 30  : depth = 0.6
    else           : depth = 0.3

    # 4. Breadth score — topic breadth
    topic_terms = [
        "cause","symptom","diagnos","treat",
        "prognos","prevent","stage","survival"
    ]
    breadth = min(
        1.0,
        sum(1 for t in topic_terms if t in a_lower) / 3.0
    )

    knowledge_coverage = round(
        reference_coverage * 0.35 +
        concept_coverage   * 0.30 +
        depth              * 0.20 +
        breadth            * 0.15,
        4
    )

    return {
        "knowledge_coverage_score": knowledge_coverage,
        "reference_coverage"      : round(reference_coverage, 4),
        "concept_coverage"        : round(concept_coverage,   4),
        "depth_score"             : round(depth,              4),
        "breadth_score"           : round(breadth,            4),
        "concepts_covered"        : covered_concepts,
        "total_concepts"          : total_concepts,
    }


# ==================================================
# METRIC H — HALLUCINATION DETECTION RATE
# Fine-grained hallucination analysis
# ==================================================

def calc_hallucination_analysis(
    answer    : str,
    reference : str,
    chunks    : list,
    model
) -> dict:

    a_emb = model.encode(
        answer[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 1. Chunk grounding
    chunk_scores = []
    for chunk in chunks[:5]:
        c_emb = model.encode(
            chunk["text"][:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        chunk_scores.append(float(np.dot(a_emb, c_emb)))

    max_chunk   = max(chunk_scores) if chunk_scores else 0.0
    mean_chunk  = (
        float(np.mean(chunk_scores))
        if chunk_scores else 0.0
    )

    # 2. Reference alignment
    r_emb = model.encode(
        reference[:500],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    ref_align = float(np.dot(a_emb, r_emb))

    # 3. Numerical claim check
    # Specific numbers without context = risk
    num_pattern = re.compile(r'\b\d+[\.,]?\d*%?\b')
    nums_in_answer    = num_pattern.findall(answer)
    nums_in_reference = num_pattern.findall(reference)
    nums_in_chunks    = num_pattern.findall(
        " ".join([c["text"] for c in chunks[:3]])
    )

    unverified_nums = [
        n for n in nums_in_answer
        if n not in nums_in_reference
        and n not in nums_in_chunks
    ]
    num_hallucination_risk = min(
        1.0, len(unverified_nums) * 0.1
    )

    # 4. Named entity verification
    # Check if specific drug/treatment names are verified
    drug_names = [
        "cisplatin","carboplatin","paclitaxel",
        "docetaxel","doxorubicin","cyclophosphamide",
        "tamoxifen","herceptin","bevacizumab",
        "pembrolizumab","nivolumab","erlotinib",
        "gefitinib","imatinib","rituximab"
    ]

    ans_drugs   = [d for d in drug_names if d in answer.lower()]
    ref_context = (
        reference + " ".join([c["text"] for c in chunks[:3]])
    ).lower()
    unverified_drugs = [
        d for d in ans_drugs if d not in ref_context
    ]
    drug_hallucination_risk = min(
        1.0, len(unverified_drugs) * 0.2
    )

    # 5. Overall hallucination score
    grounding_score = (
        max_chunk  * 0.40 +
        mean_chunk * 0.30 +
        ref_align  * 0.30
    )

    total_risk = (
        num_hallucination_risk  * 0.30 +
        drug_hallucination_risk * 0.40 +
        (1 - grounding_score)   * 0.30
    )

    hallucination_free_score = round(
        max(0.0, 1.0 - total_risk), 4
    )

    return {
        "hallucination_free_score" : hallucination_free_score,
        "grounding_score"          : round(grounding_score,         4),
        "max_chunk_similarity"     : round(max_chunk,               4),
        "mean_chunk_similarity"    : round(mean_chunk,              4),
        "ref_alignment"            : round(ref_align,               4),
        "unverified_numbers"       : len(unverified_nums),
        "unverified_drug_names"    : len(unverified_drugs),
        "hallucination_risk"       : round(total_risk,              4),
        "risk_level"               : (
            "LOW"    if total_risk < 0.2 else
            "MEDIUM" if total_risk < 0.5 else
            "HIGH"
        ),
    }


# ==================================================
# METRIC I — DISAMBIGUATION SUCCESS RATE
# Can the agent handle ambiguous questions?
# ==================================================

def calc_disambiguation(
    question : str,
    answer   : str,
    model
) -> dict:

    q_lower = question.lower()
    a_lower = answer.lower()

    # 1. Ambiguity detection in question
    ambiguous_pronouns = [
        "it","this","that","they","these",
        "those","the cancer","the disease"
    ]
    has_ambiguity = any(
        p in q_lower for p in ambiguous_pronouns
    )

    # 2. Vague question check
    vague_phrases = [
        "tell me more", "explain",
        "what about", "and the", "more info"
    ]
    is_vague = any(
        p in q_lower for p in vague_phrases
    )

    # 3. Did answer resolve ambiguity?
    specificity_phrases = [
        "specifically","in particular",
        "referring to","means","refers to",
        "in this context","to clarify"
    ]
    resolved_ambiguity = any(
        p in a_lower for p in specificity_phrases
    )

    # 4. Cancer type specificity
    cancer_types = [
        "lung","breast","colon","prostate","liver",
        "brain","cervical","ovarian","pancreatic",
        "thyroid","bladder","stomach","leukemia",
        "lymphoma","melanoma","sarcoma"
    ]
    q_has_cancer = any(c in q_lower for c in cancer_types)
    a_has_cancer = any(c in a_lower for c in cancer_types)

    cancer_specific = q_has_cancer and a_has_cancer

    # 5. Disambiguation score
    if not has_ambiguity and not is_vague:
        # Question was clear — check if answer is specific
        disam_score = 0.85 if cancer_specific else 0.70
    elif has_ambiguity or is_vague:
        # Question was ambiguous — did agent handle it?
        disam_score = (
            0.90 if resolved_ambiguity else
            0.60 if cancer_specific    else
            0.40
        )
    else:
        disam_score = 0.70

    return {
        "disambiguation_score"  : round(disam_score,   4),
        "question_was_ambiguous": has_ambiguity or is_vague,
        "ambiguity_resolved"    : resolved_ambiguity,
        "cancer_type_specific"  : cancer_specific,
    }


# ==================================================
# METRIC J — MULTI-TURN COHERENCE
# Is the agent coherent across conversation turns?
# ==================================================

def calc_multiturn_coherence(
    conversation_history : list,
    model
) -> dict:

    if len(conversation_history) < 2:
        return {
            "multiturn_coherence_score": 1.0,
            "topic_consistency"        : 1.0,
            "context_retention"        : 1.0,
            "turns_evaluated"          : len(conversation_history),
        }

    topic_scores      = []
    retention_scores  = []

    for i in range(1, len(conversation_history)):

        prev = conversation_history[i-1]
        curr = conversation_history[i]

        prev_emb = model.encode(
            prev[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        curr_emb = model.encode(
            curr[:400],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        topic_sim = float(np.dot(prev_emb, curr_emb))
        topic_scores.append(topic_sim)

        # Context retention — do later turns reference earlier?
        prev_terms = set(prev.lower().split())
        curr_terms = set(curr.lower().split())
        stops      = {
            "the","a","an","is","are","was",
            "were","in","on","at","to","of"
        }
        overlap    = (
            (prev_terms - stops) & (curr_terms - stops)
        )
        retention  = min(
            1.0, len(overlap) / max(len(prev_terms-stops), 1)
        )
        retention_scores.append(retention)

    topic_consistency = (
        float(np.mean(topic_scores))
        if topic_scores else 1.0
    )
    context_retention = (
        float(np.mean(retention_scores))
        if retention_scores else 1.0
    )

    coherence_score = round(
        topic_consistency * 0.60 +
        context_retention * 0.40,
        4
    )

    return {
        "multiturn_coherence_score": coherence_score,
        "topic_consistency"        : round(topic_consistency, 4),
        "context_retention"        : round(context_retention, 4),
        "turns_evaluated"          : len(conversation_history),
    }


# ==================================================
# EVALUATE SINGLE — ALL EXTENDED METRICS
# ==================================================

def evaluate_extended(
    qa_item    : dict,
    collection,
    model,
    history    : list = None
) -> dict:

    question   = qa_item["q"]
    reference  = qa_item["a"]
    category   = qa_item.get("category",   "general")
    difficulty = qa_item.get("difficulty", "moderate")

    # Retrieve
    chunks  = get_chunks(question, collection, model)
    context = "\n\n".join([c["text"] for c in chunks])

    # Generate
    answer = generate_answer(question, context)

    # ── All 10 Extended Metrics ───────────────────

    # A
    agent_eff = calc_agent_efficiency(
        question, answer, chunks, 1, model
    )
    # B
    query_res = calc_query_resolution(
        question, answer, model
    )
    # C
    ctx_util  = calc_context_utilization(
        answer, chunks, model
    )
    # D
    consistency = calc_response_consistency(
        question, answer, reference, model
    )
    # E
    empathy = calc_empathy_score(answer)

    # F
    safety  = calc_medical_safety(question, answer)

    # G
    knowledge = calc_knowledge_coverage(
        question, answer, reference, model
    )
    # H
    hallucination = calc_hallucination_analysis(
        answer, reference, chunks, model
    )
    # I
    disambiguation = calc_disambiguation(
        question, answer, model
    )
    # J
    conv_history = history if history else [
        question, answer
    ]
    coherence = calc_multiturn_coherence(
        conv_history, model
    )

    return {
        "id"            : qa_item["id"],
        "question"      : question,
        "answer"        : answer,
        "reference"     : reference,
        "category"      : category,
        "difficulty"    : difficulty,

        # 10 Extended metrics
        "agent_efficiency"    : agent_eff,
        "query_resolution"    : query_res,
        "context_utilization" : ctx_util,
        "response_consistency": consistency,
        "empathy"             : empathy,
        "medical_safety"      : safety,
        "knowledge_coverage"  : knowledge,
        "hallucination"       : hallucination,
        "disambiguation"      : disambiguation,
        "multiturn_coherence" : coherence,
    }


# ==================================================
# AGGREGATE ALL EXTENDED METRICS
# ==================================================

def aggregate_extended(results: list) -> dict:

    def avg(section, key):
        vals = [
            r[section][key] for r in results
            if key in r.get(section, {})
        ]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "total_evaluated"      : len(results),
        "evaluation_timestamp" : datetime.now().isoformat(),

        "A_agent_efficiency"      : {
            "agent_efficiency_score": avg("agent_efficiency","agent_efficiency_score"),
            "answer_quality"        : avg("agent_efficiency","answer_quality"),
            "avg_chunk_relevance"   : avg("agent_efficiency","avg_chunk_relevance"),
            "chunk_utilization"     : avg("agent_efficiency","chunk_utilization"),
            "length_efficiency"     : avg("agent_efficiency","length_efficiency"),
        },
        "B_query_resolution"      : {
            "query_resolution_rate" : avg("query_resolution","query_resolution_rate"),
            "semantic_match"        : avg("query_resolution","semantic_match"),
            "term_coverage"         : avg("query_resolution","term_coverage"),
            "resolution_density"    : avg("query_resolution","resolution_density"),
        },
        "C_context_utilization"   : {
            "context_utilization_score": avg("context_utilization","context_utilization_score"),
            "coverage_depth"           : avg("context_utilization","coverage_depth"),
            "source_diversity"         : avg("context_utilization","source_diversity"),
            "context_integration"      : avg("context_utilization","context_integration"),
        },
        "D_response_consistency"  : {
            "response_consistency_score": avg("response_consistency","response_consistency_score"),
            "qa_consistency"            : avg("response_consistency","qa_consistency"),
            "ar_consistency"            : avg("response_consistency","ar_consistency"),
            "term_consistency"          : avg("response_consistency","term_consistency"),
            "specificity"               : avg("response_consistency","specificity"),
        },
        "E_empathy"               : {
            "empathy_score"        : avg("empathy","empathy_score"),
            "reassurance_score"    : avg("empathy","reassurance_score"),
            "acknowledgment_score" : avg("empathy","acknowledgment_score"),
            "hope_score"           : avg("empathy","hope_score"),
            "hedging_score"        : avg("empathy","hedging_score"),
        },
        "F_medical_safety"        : {
            "medical_safety_score"  : avg("medical_safety","medical_safety_score"),
            "has_disclaimer_rate"   : round(float(np.mean([
                1.0 if r["medical_safety"]["has_disclaimer"] else 0.0
                for r in results
            ])), 4),
            "has_evidence_rate"     : round(float(np.mean([
                1.0 if r["medical_safety"]["has_evidence_language"] else 0.0
                for r in results
            ])), 4),
            "in_scope_rate"         : round(float(np.mean([
                1.0 if r["medical_safety"]["in_oncology_scope"] else 0.0
                for r in results
            ])), 4),
        },
        "G_knowledge_coverage"    : {
            "knowledge_coverage_score": avg("knowledge_coverage","knowledge_coverage_score"),
            "reference_coverage"      : avg("knowledge_coverage","reference_coverage"),
            "concept_coverage"        : avg("knowledge_coverage","concept_coverage"),
            "depth_score"             : avg("knowledge_coverage","depth_score"),
            "breadth_score"           : avg("knowledge_coverage","breadth_score"),
        },
        "H_hallucination"         : {
            "hallucination_free_score": avg("hallucination","hallucination_free_score"),
            "grounding_score"         : avg("hallucination","grounding_score"),
            "mean_chunk_similarity"   : avg("hallucination","mean_chunk_similarity"),
            "ref_alignment"           : avg("hallucination","ref_alignment"),
            "low_risk_rate"           : round(float(np.mean([
                1.0 if r["hallucination"]["risk_level"] == "LOW" else 0.0
                for r in results
            ])), 4),
        },
        "I_disambiguation"        : {
            "disambiguation_score": avg("disambiguation","disambiguation_score"),
            "cancer_specific_rate": round(float(np.mean([
                1.0 if r["disambiguation"]["cancer_type_specific"] else 0.0
                for r in results
            ])), 4),
        },
        "J_multiturn_coherence"   : {
            "multiturn_coherence_score": avg("multiturn_coherence","multiturn_coherence_score"),
            "topic_consistency"        : avg("multiturn_coherence","topic_consistency"),
            "context_retention"        : avg("multiturn_coherence","context_retention"),
        },
    }


# ==================================================
# PRINT EXTENDED REPORT
# ==================================================

def print_extended_report(summary: dict):

    print(f"\n{'='*65}")
    print(f"  NITPY — EXTENDED AGENT EVALUATION REPORT")
    print(f"  10 New Metrics Beyond Standard NLP Scores")
    print(f"{'='*65}")
    print(f"  Questions evaluated : {summary['total_evaluated']}")
    print()

    sections = [
        ("A", "Agent Efficiency",
         "A_agent_efficiency",
         "agent_efficiency_score"),
        ("B", "Query Resolution Rate",
         "B_query_resolution",
         "query_resolution_rate"),
        ("C", "Context Utilization",
         "C_context_utilization",
         "context_utilization_score"),
        ("D", "Response Consistency",
         "D_response_consistency",
         "response_consistency_score"),
        ("E", "Empathy Score",
         "E_empathy",
         "empathy_score"),
        ("F", "Medical Safety",
         "F_medical_safety",
         "medical_safety_score"),
        ("G", "Knowledge Coverage",
         "G_knowledge_coverage",
         "knowledge_coverage_score"),
        ("H", "Hallucination Free Rate",
         "H_hallucination",
         "hallucination_free_score"),
        ("I", "Disambiguation Score",
         "I_disambiguation",
         "disambiguation_score"),
        ("J", "Multi-turn Coherence",
         "J_multiturn_coherence",
         "multiturn_coherence_score"),
    ]

    print(f"  {'─'*60}")
    print(f"  {'Metric':<35} {'Score':>10}  Status")
    print(f"  {'─'*60}")

    all_scores = []
    for code, name, section, key in sections:
        score  = summary[section][key]
        target = 0.75
        status = "✅" if score >= target else "⚠️"
        bar    = "█" * int(score * 20)
        all_scores.append(score)
        print(
            f"  {code}. {name:<33} "
            f"{score:>8.4f}  {status}"
        )

    print(f"  {'─'*60}")
    print(
        f"  {'Overall Agent Score':<35} "
        f"{round(float(np.mean(all_scores)),4):>8.4f}  "
        f"{'✅' if np.mean(all_scores) >= 0.75 else '⚠️'}"
    )
    print(f"  {'─'*60}")

    # Detailed breakdown
    print(f"\n  {'─'*60}")
    print(f"  DETAILED BREAKDOWN")
    print(f"  {'─'*60}")

    for code, name, section, key in sections:
        data = summary[section]
        print(f"\n  [{code}] {name}")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"      {k:<35} : {v:.4f}")
            else:
                print(f"      {k:<35} : {v}")

    print(f"\n{'='*65}")

    # Save
    os.makedirs("results", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/eval_EXTENDED_{ts}.json"

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Report saved → {path}\n")
    return path


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import nltk

    print("\nDownloading NLTK...")
    for pkg in [
        "punkt","punkt_tab","wordnet",
        "omw-1.4","averaged_perceptron_tagger"
    ]:
        nltk.download(pkg, quiet=True)
    print("  NLTK ready ✅")

    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Model loaded ✅")

    print("\nLoading ChromaDB...")
    import chromadb

    client     = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name     = "medical_rag",
        metadata = {"hnsw:space": "cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading QA data...")
    qa_data = load_qa_data("data/cleaned_output.json")

    print(f"\nEvaluating {len(qa_data)} questions...")
    print("="*65)

    all_results  = []
    conv_history = []

    for i, qa_item in enumerate(qa_data):

        print(f"[{i+1}/{len(qa_data)}] {qa_item['q'][:55]}...")

        try:
            result = evaluate_extended(
                qa_item    = qa_item,
                collection = collection,
                model      = model,
                history    = conv_history[-6:]
            )
            all_results.append(result)

            # Update conversation history
            conv_history.append(qa_item["q"])
            conv_history.append(result["answer"])

            # Keep last 6 turns
            if len(conv_history) > 12:
                conv_history = conv_history[-12:]

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nAggregating {len(all_results)} results...")
    summary = aggregate_extended(all_results)
    print_extended_report(summary)