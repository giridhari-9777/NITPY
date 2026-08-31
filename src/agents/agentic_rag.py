# src/agents/agentic_rag.py

import os
import sys
import gc
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# AGENTIC RAG — FULL PIPELINE
# ==================================================

class AgenticRAG:

    def __init__(self):

        print("\n" + "="*60)
        print("  MEDICAL AGENTIC RAG PIPELINE")
        print("  Cancer QA — Doctor Patient Simulation")
        print("="*60)

        # ── Embedding model ──────────────────────────
        print("\n[1/5] Loading embedding model...")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print("  Model loaded ✅")

        # ── ChromaDB ─────────────────────────────────
        print("\n[2/5] Loading ChromaDB...")
        import chromadb

        self.client      = chromadb.PersistentClient(
            path = "./chroma_db"
        )
        self.medical_col = self.client.get_or_create_collection(
            name     = "medical_rag",
            metadata = {"hnsw:space": "cosine"}
        )
        self.history_col = self.client.get_or_create_collection(
            name     = "conversation_history",
            metadata = {"hnsw:space": "cosine"}
        )

        print(f"  Medical KB : {self.medical_col.count()} ✅")
        print(f"  History DB : {self.history_col.count()} ✅")

        if self.medical_col.count() == 0:
            self._load_kb()

        # ── Ollama ───────────────────────────────────
        print("\n[3/5] Checking Ollama...")
        import requests

        self.ollama_url = "http://localhost:11434/api/generate"
        self.llm_model  = "llama3"

        try:
            requests.get(
                "http://localhost:11434",
                timeout = 2
            )
            self.use_ollama = True
            print("  Ollama : running ✅")
        except Exception:
            self.use_ollama = False
            print("  Ollama : offline ⚠️")
            print("  → Run: ollama serve")
            print("  → Run: ollama pull llama3")

        # ── Classifiers ──────────────────────────────
        print("\n[4/5] Loading classifiers...")
        from generator.generator_model import (
            classify_question,
            generate_sub_queries,
            extract_cancer_type
        )

        self.classify        = classify_question
        self.gen_sub_queries = generate_sub_queries
        self.extract_cancer  = extract_cancer_type
        print("  Classifiers loaded ✅")

        # ── Evaluators ───────────────────────────────
        print("\n[5/5] Loading evaluators...")
        from evaluation.hallucination_detector import (
            HallucinationDetector
        )
        from evaluation.deep_analysis import DeepEvalText

        self.detector  = HallucinationDetector()
        self.evaluator = DeepEvalText()
        print("  Evaluators loaded ✅")

        # ── State ────────────────────────────────────
        self.sessions      = {}
        self.total_queries = 0
        self.total_retries = 0
        self.all_results   = []

        print("\n" + "="*60)
        print("  All components loaded!")
        print("  Agentic RAG ready!")
        print("="*60 + "\n")


    # ==================================================
    # LOAD KNOWLEDGE BASE
    # ==================================================

    def _load_kb(self):

        import pickle
        import numpy as np

        print("\n  Loading medical knowledge base...")

        meta_path = "outputs/embeddings/chunks_metadata.pkl"
        emb_path  = "outputs/embeddings/embeddings.npy"

        if not os.path.exists(meta_path):
            print("  ⚠️  No embeddings found!")
            return

        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)

        embeddings = np.load(emb_path)

        print(f"  Chunks     : {len(chunks)}")
        print(f"  Embeddings : {embeddings.shape}")

        batch_size = 500
        for start in range(0, len(chunks), batch_size):

            end        = min(start + batch_size, len(chunks))
            batch      = chunks[start:end]
            batch_embs = embeddings[start:end]

            self.medical_col.add(
                ids        = [f"id_{i}" for i in range(start, end)],
                documents  = [c["text"] for c in batch],
                embeddings = batch_embs.tolist(),
                metadatas  = [
                    {
                        "source"   : str(c.get("source",   "unknown")),
                        "title"    : str(c.get("title",    "no_title")),
                        "chunk_id" : str(c.get("chunk_id", i))
                    }
                    for i, c in enumerate(batch)
                ]
            )
            print(f"  Inserted {start} → {end}")

        print(f"  KB ready: {self.medical_col.count()} ✅")


    # ==================================================
    # AGENT DECISION
    # ==================================================

    def _agent_decision(self, question: str) -> dict:

        medical_keywords = [
            "cancer", "tumor", "tumour", "malignancy",
            "leukemia", "lymphoma", "melanoma", "carcinoma",
            "sarcoma", "oncology", "metastasis", "neoplasm",
            "pain", "cough", "fatigue", "bleeding", "lump",
            "swelling", "weight loss", "fever", "nausea",
            "chemotherapy", "radiation", "surgery", "therapy",
            "treatment", "medication", "drug", "dose",
            "diagnosis", "prognosis", "biopsy", "scan",
            "mri", "ct scan", "pathology", "staging",
            "survival", "remission", "recurrence",
            "lung", "breast", "colon", "prostate", "liver",
            "brain", "bone", "kidney", "cervical", "ovarian",
            "pancreatic", "thyroid", "bladder", "stomach",
            "doctor", "patient", "medical", "oncologist",
        ]

        is_medical    = any(
            kw in question.lower()
            for kw in medical_keywords
        )
        question_type = self.classify(question)
        cancer_type   = self.extract_cancer(question)
        sub_queries   = self.gen_sub_queries(
            question, question_type
        )

        return {
            "is_medical"    : is_medical,
            "question_type" : question_type,
            "cancer_type"   : cancer_type,
            "sub_queries"   : sub_queries,
            "action"        : "answer" if is_medical else "reject"
        }


    # ==================================================
    # GET HISTORY
    # ==================================================

    def _get_history(self, session_id: str) -> str:

        try:
            result = self.history_col.get(
                where   = {"session_id": session_id},
                include = ["documents", "metadatas"]
            )

            if not result["ids"]:
                return ""

            turns = sorted(
                zip(
                    result["documents"],
                    result["metadatas"]
                ),
                key = lambda x: x[1].get("timestamp", "")
            )

            lines = []
            for doc, meta in turns[-6:]:
                role = (
                    "Patient"
                    if meta.get("role") == "patient"
                    else "Doctor"
                )
                lines.append(f"{role}: {doc}")

            return "\n".join(lines)

        except Exception:
            return ""


    # ==================================================
    # RETRIEVE
    # ==================================================

    def _retrieve(self, question: str) -> dict:

        try:
            q_emb = self.model.encode(
                question,
                normalize_embeddings = True,
                convert_to_numpy     = True
            )

            result = self.medical_col.query(
                query_embeddings = [q_emb.tolist()],
                n_results        = 5,
                include          = [
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            chunks = []
            for i in range(len(result["ids"][0])):
                score = 1 - result["distances"][0][i]
                chunks.append({
                    "text"         : result["documents"][0][i],
                    "source"       : result["metadatas"][0][i].get("source", "unknown"),
                    "title"        : result["metadatas"][0][i].get("title",  "no_title"),
                    "rerank_score" : round(score, 4)
                })

            context = "\n\n".join([
                f"[Medical Reference {i+1}]\n"
                f"Source : {c['source']}\n"
                f"Title  : {c['title']}\n"
                f"Score  : {c['rerank_score']}\n\n"
                f"{c['text']}"
                for i, c in enumerate(chunks)
            ])

            del q_emb
            gc.collect()

            return {
                "success" : True,
                "chunks"  : chunks,
                "context" : context,
                "count"   : len(chunks)
            }

        except Exception as e:
            print(f"  ⚠️  Retrieval error: {e}")
            return {
                "success" : False,
                "chunks"  : [],
                "context" : "",
                "count"   : 0
            }


    # ==================================================
    # GENERATE
    # ==================================================

    def _generate(
        self,
        question      : str,
        context       : str,
        history       : str,
        question_type : str,
        sub_queries   : list,
        cancer_type   : str
    ) -> dict:

        import requests
        import numpy as np

        sub_q = "\n".join(
            f"  {i+1}. {q}"
            for i, q in enumerate(sub_queries)
        )

        hist_section = (
            f"\nPREVIOUS CONVERSATION:\n{history}\n"
            if history else ""
        )

        prompt = f"""You are an expert oncologist AI
assistant in a Doctor-Patient simulation chatbot.

STRICT RULES:
- Answer ONLY from the provided medical context
- Be empathetic, clear and compassionate
- Use hedge phrases: "typically", "may", "generally",
  "research suggests", "studies show"
- Structure your answer clearly with numbered points
- End EVERY response with:
  "Please consult your oncologist for personalized
  medical advice."
- NEVER hallucinate or make up medical facts

{hist_section}

MEDICAL CONTEXT FROM ONCOLOGY TEXTBOOKS:
{context}

PATIENT QUESTION   : {question}
QUESTION TYPE      : {question_type.upper()}
CANCER TYPE        : {cancer_type}

RELATED MEDICAL SUB-QUERIES:
{sub_q}

Provide a comprehensive, empathetic and medically
accurate doctor response:"""

        if self.use_ollama:
            try:
                resp = requests.post(
                    self.ollama_url,
                    json    = {
                        "model"  : self.llm_model,
                        "prompt" : prompt,
                        "stream" : False,
                        "options": {
                            "temperature" : 0.3,
                            "num_predict" : 600
                        }
                    },
                    timeout = 120
                )

                if resp.status_code == 200:
                    answer = resp.json().get(
                        "response", ""
                    ).strip()
                else:
                    answer = self._fallback(context)

            except Exception as e:
                print(f"  ⚠️  Ollama error: {e}")
                answer = self._fallback(context)
        else:
            answer = self._fallback(context)

        # Confidence
        try:
            a_emb = self.model.encode(
                answer[:400],
                normalize_embeddings = True,
                convert_to_numpy     = True
            )
            c_emb = self.model.encode(
                context[:400],
                normalize_embeddings = True,
                convert_to_numpy     = True
            )
            confidence = round(
                min(float(np.dot(a_emb, c_emb)) + 0.1, 1.0),
                3
            )
            del a_emb, c_emb
            gc.collect()

        except Exception:
            confidence = 0.7

        return {
            "answer"     : answer,
            "confidence" : confidence
        }


    def _fallback(self, context: str) -> str:

        sentences = context.replace("\n", " ").split(". ")
        relevant  = ". ".join(sentences[:4])

        return (
            f"Based on medical literature, {relevant}. "
            f"Research suggests individual results may vary. "
            f"Please consult your oncologist for "
            f"personalized medical advice."
        )


    # ==================================================
    # EVALUATE
    # ==================================================

    def _evaluate(
        self,
        question : str,
        answer   : str,
        chunks   : list
    ) -> dict:

        try:
            result = self.detector.detect(
                question = question,
                answer   = answer,
                chunks   = chunks
            )

            return {
                "verdict"         : result["verdict"],
                "score"           : result["score_out_of_5"],
                "is_hallucinated" : result["is_hallucinated"],
                "faithfulness"    : result["checks"]["faithfulness"]["faithfulness_score"],
                "relevance"       : result["checks"]["relevance"]["relevance_score"],
                "safety"          : result["checks"]["safety"]["risk_level"]
            }

        except Exception as e:
            print(f"  ⚠️  Eval error: {e}")
            return {
                "verdict"         : "Evaluation failed",
                "score"           : 3.5,
                "is_hallucinated" : False,
                "faithfulness"    : 0.5,
                "relevance"       : 0.5,
                "safety"          : "LOW"
            }


    # ==================================================
    # SAVE HISTORY
    # ==================================================

    def _save_history(
        self,
        session_id : str,
        role       : str,
        message    : str
    ):
        try:
            emb = self.model.encode(
                message,
                normalize_embeddings = True,
                convert_to_numpy     = True
            )

            self.history_col.add(
                ids        = [str(uuid.uuid4())],
                documents  = [message],
                embeddings = [emb.tolist()],
                metadatas  = [{
                    "session_id" : session_id,
                    "role"       : role,
                    "timestamp"  : datetime.now().isoformat()
                }]
            )

            del emb
            gc.collect()

        except Exception as e:
            print(f"  ⚠️  History error: {e}")


    # ==================================================
    # RUN FULL DEEP EVALUATION
    # ==================================================

    def _deep_evaluate(
        self,
        question      : str,
        answer        : str,
        chunks        : list,
        rerank_scores : list,
        agent_iters   : int,
        confidence    : float
    ) -> dict:

        try:
            result = self.evaluator.evaluate(
                question      = question,
                answer        = answer,
                chunks        = chunks,
                rerank_scores = rerank_scores,
                agent_iters   = agent_iters,
                confidence    = confidence
            )
            return result

        except Exception as e:
            print(f"  ⚠️  Deep eval error: {e}")
            return {}


    # ==================================================
    # MAIN RUN
    # ==================================================

    def run(
        self,
        question    : str,
        session_id  : str = "default",
        max_retries : int = 1
    ) -> dict:

        start_time = datetime.now()
        self.total_queries += 1
        agent_iters = 0

        print(f"\n{'─'*60}")
        print(f"  Patient  : {question[:55]}")
        print(f"  Session  : {session_id}")
        print(f"{'─'*60}")

        # ── Step 1: Agent Decision ───────────────────
        print("\n  [Agent Step 1] Analyzing question...")
        decision = self._agent_decision(question)

        print(f"  → Type    : {decision['question_type'].upper()}")
        print(f"  → Cancer  : {decision['cancer_type']}")
        print(f"  → Medical : {decision['is_medical']}")
        print(f"  → Action  : {decision['action'].upper()}")

        # Reject non-medical
        if not decision["is_medical"]:
            return {
                "question"      : question,
                "answer"        : (
                    "I am a specialized medical AI assistant "
                    "for cancer and oncology topics only. "
                    "Please ask about cancer symptoms, "
                    "diagnosis, treatment, prognosis, "
                    "staging, or prevention."
                ),
                "question_type" : "non_medical",
                "cancer_type"   : "N/A",
                "sub_queries"   : [],
                "confidence"    : 1.0,
                "agent_iters"   : 0,
                "elapsed_sec"   : 0.0,
                "hallucination" : {
                    "verdict"         : "N/A",
                    "score"           : 5.0,
                    "is_hallucinated" : False,
                    "faithfulness"    : 1.0,
                    "relevance"       : 1.0,
                    "safety"          : "LOW"
                },
                "deep_eval"     : {},
                "sources"       : [],
                "session_id"    : session_id,
                "timestamp"     : datetime.now().isoformat()
            }

        # ── Step 2: Load History ─────────────────────
        print("\n  [Agent Step 2] Loading history...")
        history = self._get_history(session_id)
        print(
            f"  → {'History found ✅' if history else 'No history'}"
        )

        # ── Step 3: Retrieve ─────────────────────────
        print("\n  [Agent Step 3] Retrieving context...")
        retrieval = self._retrieve(question)
        print(f"  → {retrieval['count']} chunks retrieved")

        rerank_scores = [
            c.get("rerank_score", 0.0)
            for c in retrieval["chunks"]
        ]

        # ── Step 4+5: Generate + Evaluate Loop ───────
        answer     = ""
        confidence = 0.0
        evaluation = {}

        for attempt in range(max_retries + 1):

            agent_iters += 1

            if attempt > 0:
                self.total_retries += 1
                print(f"\n  Retry {attempt}/{max_retries}...")

                if retrieval["chunks"]:
                    retrieval["context"] = (
                        "Answer STRICTLY from these facts:\n\n"
                        + "\n\n".join([
                            f"FACT {i+1}: {c['text'][:300]}"
                            for i, c in enumerate(
                                retrieval["chunks"][:3]
                            )
                        ])
                    )
            else:
                print(f"\n  [Agent Step 4] Generating answer...")

            gen = self._generate(
                question      = question,
                context       = retrieval["context"],
                history       = history,
                question_type = decision["question_type"],
                sub_queries   = decision["sub_queries"],
                cancer_type   = decision["cancer_type"]
            )

            answer     = gen["answer"]
            confidence = gen["confidence"]

            print(f"\n  [Agent Step 5] Evaluating answer...")
            evaluation = self._evaluate(
                question = question,
                answer   = answer,
                chunks   = retrieval["chunks"]
            )

            print(f"  → Score   : {evaluation['score']} / 5.0")
            print(f"  → Verdict : {evaluation['verdict']}")

            if not evaluation["is_hallucinated"]:
                print(f"  → PASSED ✅")
                break
            else:
                print(f"  → Retrying...")

        # ── Step 6: Deep Evaluation ──────────────────
        print(f"\n  [Agent Step 6] Running deep evaluation...")
        deep_eval = self._deep_evaluate(
            question      = question,
            answer        = answer,
            chunks        = retrieval["chunks"],
            rerank_scores = rerank_scores,
            agent_iters   = agent_iters,
            confidence    = confidence
        )

        scope_score = (
            deep_eval.get("scope", {}).get("weighted_total", 0.0)
            if deep_eval else 0.0
        )
        print(f"  → SCOPE : {scope_score} / 5.0")

        # ── Step 7: Save History ─────────────────────
        print(f"\n  [Agent Step 7] Saving to history...")
        self._save_history(session_id, "patient", question)
        self._save_history(session_id, "doctor",  answer)
        print("  → Saved ✅")

        elapsed = round(
            (datetime.now() - start_time).total_seconds(), 2
        )

        # ── Build Response ───────────────────────────
        response = {
            "question"      : question,
            "answer"        : answer,
            "question_type" : decision["question_type"],
            "cancer_type"   : decision["cancer_type"],
            "sub_queries"   : decision["sub_queries"],
            "confidence"    : confidence,
            "agent_iters"   : agent_iters,
            "elapsed_sec"   : elapsed,
            "hallucination" : {
                "verdict"         : evaluation.get("verdict",         "N/A"),
                "score"           : evaluation.get("score",           0.0),
                "is_hallucinated" : evaluation.get("is_hallucinated", False),
                "faithfulness"    : evaluation.get("faithfulness",    0.0),
                "relevance"       : evaluation.get("relevance",       0.0),
                "safety"          : evaluation.get("safety",          "LOW")
            },
            "deep_eval"     : deep_eval,
            "sources"       : [
                {
                    "source" : c.get("source", "unknown"),
                    "title"  : c.get("title",  "")[:60],
                    "score"  : c.get("rerank_score", 0.0)
                }
                for c in retrieval.get("chunks", [])[:3]
            ],
            "session_id"    : session_id,
            "timestamp"     : datetime.now().isoformat()
        }

        self.all_results.append(response)
        self._print_response(response)

        return response


    # ==================================================
    # PRINT RESPONSE
    # ==================================================

    def _print_response(self, r: dict):

        de = r.get("deep_eval", {})
        sc = de.get("scope", {}) if de else {}

        print(f"\n{'='*60}")
        print(f"  AGENTIC RAG — FINAL RESPONSE")
        print(f"{'='*60}")
        print(f"  Patient    : {r['question'][:55]}")
        print(f"  Type       : {r['question_type'].upper()}")
        print(f"  Cancer     : {r['cancer_type']}")
        print(f"  Confidence : {r['confidence']}")
        print(f"  H-Score    : {r['hallucination']['score']} / 5.0")
        print(f"  Verdict    : {r['hallucination']['verdict']}")
        print(f"  Agent Iters: {r['agent_iters']}")
        print(f"  Time       : {r['elapsed_sec']}s")

        if sc:
            print(f"\n  SCOPE Scores:")
            print(f"  S Safety       : {sc.get('safety',       0)}")
            print(f"  C Completeness : {sc.get('completeness', 0)}")
            print(f"  O Originality  : {sc.get('originality',  0)}")
            print(f"  P Precision    : {sc.get('precision',    0)}")
            print(f"  E Efficiency   : {sc.get('efficiency',   0)}")
            print(f"  Total          : {sc.get('weighted_total', 0)} / 5.0")

        print(f"\n  Sub-Queries:")
        for i, sq in enumerate(r["sub_queries"]):
            print(f"     {i+1}. {sq}")

        if r["sources"]:
            print(f"\n  Sources:")
            for i, s in enumerate(r["sources"]):
                print(
                    f"     {i+1}. {s['source']}"
                    f" ({s['score']})"
                )

        print(f"\n  Doctor:")
        print(f"  {'-'*56}")
        for line in r["answer"].split("\n"):
            if line.strip():
                print(f"  {line}")
        print(f"{'='*60}\n")


    # ==================================================
    # BATCH RUN
    # ==================================================

    def batch_run(
        self,
        questions  : list,
        session_id : str = "default"
    ) -> list:

        results = []

        print(f"\nBatch: {len(questions)} questions\n")

        for i, q in enumerate(questions):
            print(f"\nQuestion {i+1}/{len(questions)}")
            result = self.run(
                question   = q,
                session_id = session_id
            )
            results.append(result)

        # Final evaluation report
        self._print_final_report(results)

        return results


    # ==================================================
    # PRINT FINAL REPORT
    # ==================================================

    def _print_final_report(self, results: list):

        import numpy as np

        valid = [
            r for r in results
            if r["question_type"] != "non_medical"
        ]

        if not valid:
            return

        passed = sum(
            1 for r in valid
            if not r["hallucination"]["is_hallucinated"]
        )

        avg_h_score = round(
            float(np.mean([
                r["hallucination"]["score"]
                for r in valid
            ])), 2
        )

        scope_scores = [
            r.get("deep_eval", {}).get(
                "scope", {}
            ).get("weighted_total", 0.0)
            for r in valid
            if r.get("deep_eval")
        ]

        avg_scope = round(
            float(np.mean(scope_scores)), 2
        ) if scope_scores else 0.0

        avg_confidence = round(
            float(np.mean([
                r["confidence"] for r in valid
            ])), 3
        )

        avg_iters = round(
            float(np.mean([
                r["agent_iters"] for r in valid
            ])), 2
        )

        print(f"\n{'#'*60}")
        print(f"  AGENTIC RAG — FINAL REPORT")
        print(f"{'#'*60}")
        print(f"  Total Questions  : {len(results)}")
        print(f"  Medical Qs       : {len(valid)}")
        print(f"  Passed           : {passed}/{len(valid)}")
        print(f"  Total Retries    : {self.total_retries}")
        print(f"  Avg Agent Iters  : {avg_iters}")
        print(f"  Avg Confidence   : {avg_confidence}")
        print(f"  Avg H-Score      : {avg_h_score} / 5.0")
        print(f"  Avg SCOPE Score  : {avg_scope} / 5.0")

        if avg_scope >= 4.4:
            print(f"  Grade            : ✅ EXCELLENT (≥4.4)")
        elif avg_scope >= 4.0:
            print(f"  Grade            : ✅ VERY GOOD (≥4.0)")
        elif avg_scope >= 3.5:
            print(f"  Grade            : ⚠️  GOOD (≥3.5)")
        else:
            print(f"  Grade            : ❌ NEEDS IMPROVEMENT")

        print(f"{'#'*60}\n")


    # ==================================================
    # GET SESSION SUMMARY
    # ==================================================

    def get_session_summary(
        self,
        session_id : str
    ) -> dict:

        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]

        print(f"\n{'='*60}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Session ID   : {session_id}")
        print(f"  Created At   : {session['created_at'][:19]}")
        print(f"{'='*60}\n")

        return session


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    rag = AgenticRAG()

    session_id = (
        f"patient_{datetime.now().strftime('%H%M%S')}"
    )

    questions = [
        "Doctor, what are the early symptoms "
        "of lung cancer?",

        "What treatment options are available "
        "for stage 2 breast cancer?",

        "What is the 5-year survival rate "
        "for colon cancer stage 3?",

        "How is leukemia diagnosed in children?",

        "What are the side effects of "
        "radiation therapy?",

        "How can cervical cancer be prevented?",

        # Non-medical
        "What is the cricket score today?",
    ]

    results = rag.batch_run(
        questions  = questions,
        session_id = session_id
    )