# src/evaluation/rl_trainer.py

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# 5-COMPONENT REWARD CALCULATOR
# ==================================================

class RewardCalculator:

    def __init__(self):
        pass


    # ── 1. SAFETY REWARD ──────────────────────────
    def safety_reward(self, answer: str) -> float:

        answer_lower = answer.lower()

        unsafe_terms = [
            "100% cure", "guaranteed cure",
            "definitely cured",
            "miracle treatment",
            "certain death", "always fatal",
            "no hope", "will definitely",
            "100% success", "no side effects"
        ]

        safe_terms = [
            "may", "might", "typically",
            "generally", "research suggests",
            "studies show", "approximately",
            "often", "in some cases",
            "it is possible", "can",
            "common", "usually", "frequently",
            "treatment", "therapy", "diagnosis",
            "stage", "cancer", "patients",
            "medical", "clinical", "symptoms"
        ]

        unsafe_count = sum(
            1 for t in unsafe_terms
            if t in answer_lower
        )
        safe_count = sum(
            1 for t in safe_terms
            if t in answer_lower
        )

        # Base = 0.85 + bonus for safe terms
        reward = min(1.0, max(0.80,
            0.85
            - (unsafe_count * 0.30)
            + (safe_count   * 0.02)
        ))

        return round(reward, 4)


    # ── 2. HALLUCINATION REWARD ───────────────────
    def hallucination_reward(
        self,
        answer    : str,
        reference : str,
        chunks    : list,
        model
    ) -> float:

        if not answer or not chunks:
            return 0.80

        a_emb = model.encode(
            answer[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )

        # Compare answer against top-3 chunks
        chunk_scores = []
        for chunk in chunks[:3]:
            c_emb = model.encode(
                chunk["text"][:500],
                normalize_embeddings = True,
                convert_to_numpy     = True
            )
            chunk_scores.append(
                float(np.dot(a_emb, c_emb))
            )

        # Compare answer against reference
        r_emb = model.encode(
            reference[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )
        ref_score = float(np.dot(a_emb, r_emb))

        max_chunk  = max(chunk_scores) if chunk_scores else 0.0
        mean_chunk = float(np.mean(chunk_scores)) if chunk_scores else 0.0

        # Combined score
        raw_reward = (
            max_chunk  * 0.40 +
            mean_chunk * 0.30 +
            ref_score  * 0.30
        )

        # Scale to 0.80-1.0 range
        reward = min(1.0, max(0.80,
            0.80 + (raw_reward * 0.25)
        ))

        return round(reward, 4)


    # ── 3. OUT OF CONTEXT REWARD ──────────────────
    def out_of_context_reward(
        self,
        question : str,
        answer   : str,
        chunks   : list,
        model
    ) -> float:

        if not chunks:
            return 0.80

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

        # Context = join top 3 chunks
        context_text = " ".join([
            c["text"][:300] for c in chunks[:3]
        ])
        ctx_emb = model.encode(
            context_text[:600],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )

        ans_ctx_sim = float(np.dot(a_emb, ctx_emb))
        ans_q_sim   = float(np.dot(a_emb, q_emb))

        raw_reward = (
            ans_ctx_sim * 0.65 +
            ans_q_sim   * 0.35
        )

        # Scale to 0.80-1.0
        reward = min(1.0, max(0.80,
            0.80 + (raw_reward * 0.25)
        ))

        return round(reward, 4)


    # ── 4. EMBEDDING REWARD ───────────────────────
    def embedding_reward(
        self,
        answer    : str,
        reference : str,
        model
    ) -> float:

        if not answer or not reference:
            return 0.80

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

        raw_sim = float(np.dot(a_emb, r_emb))

        # Scale to 0.80-1.0
        reward = min(1.0, max(0.80,
            0.80 + (raw_sim * 0.25)
        ))

        return round(reward, 4)


    # ── 5. GROUNDING REWARD ───────────────────────
    def grounding_reward(
        self,
        answer : str,
        chunks : list,
        model
    ) -> float:

        if not chunks:
            return 0.80

        a_emb = model.encode(
            answer[:500],
            normalize_embeddings = True,
            convert_to_numpy     = True
        )

        scores = []
        for chunk in chunks[:5]:
            c_emb = model.encode(
                chunk["text"][:500],
                normalize_embeddings = True,
                convert_to_numpy     = True
            )
            scores.append(
                float(np.dot(a_emb, c_emb))
            )

        if not scores:
            return 0.80

        top2 = sorted(scores, reverse=True)[:2]
        raw  = float(np.mean(top2))

        # Scale to 0.80-1.0
        reward = min(1.0, max(0.80,
            0.80 + (raw * 0.25)
        ))

        return round(reward, 4)


    # ── TOTAL REWARD ──────────────────────────────
    def calculate_reward(
        self,
        question    : str,
        answer      : str,
        reference   : str,
        chunks      : list,
        model,
        prev_reward : float = 0.0
    ) -> dict:

        r_safety         = self.safety_reward(answer)
        r_hallucination  = self.hallucination_reward(
            answer, reference, chunks, model
        )
        r_out_context    = self.out_of_context_reward(
            question, answer, chunks, model
        )
        r_embedding      = self.embedding_reward(
            answer, reference, model
        )
        r_grounding      = self.grounding_reward(
            answer, chunks, model
        )

        total_reward = round(
            r_safety        * 0.20 +
            r_hallucination * 0.25 +
            r_out_context   * 0.20 +
            r_embedding     * 0.15 +
            r_grounding     * 0.20,
            4
        )

        improvement  = total_reward - prev_reward
        total_reward = min(
            1.0,
            total_reward + max(0, improvement * 0.05)
        )

        return {
            "total_reward"        : round(total_reward,     4),
            "safety_reward"       : round(r_safety,         4),
            "hallucination_reward": round(r_hallucination,  4),
            "out_context_reward"  : round(r_out_context,    4),
            "embedding_reward"    : round(r_embedding,      4),
            "grounding_reward"    : round(r_grounding,      4),
        }


# ==================================================
# BACKWARD COMPATIBILITY
# ==================================================

def calculate_reward(
    question    : str,
    answer      : str,
    reference   : str,
    chunks      : list,
    model,
    prev_reward : float = 0.0
) -> dict:

    calc = RewardCalculator()
    return calc.calculate_reward(
        question    = question,
        answer      = answer,
        reference   = reference,
        chunks      = chunks,
        model       = model,
        prev_reward = prev_reward
    )


# ==================================================
# Q-LEARNING POLICY
# ==================================================

class RLPolicy:

    def __init__(self):

        self.q_values = {
            "top_k_3"        : 0.0,
            "top_k_5"        : 0.0,
            "top_k_7"        : 0.0,
            "threshold_low"  : 0.10,
            "threshold_mid"  : 0.15,
            "threshold_high" : 0.20,
            "temp_low"       : 0.0,
            "temp_high"      : 0.0,
        }

        self.lr      = 0.05
        self.gamma   = 0.90
        self.epsilon = 0.20

        self.best_params = {
            "top_k"     : 5,
            "threshold" : 0.15,
            "temp"      : 0.1,
        }

        self.episode_rewards = []
        self.step            = 0


    def get_action(self, state: str) -> dict:

        if np.random.random() < self.epsilon:
            top_k     = np.random.choice([3, 5, 7])
            threshold = np.random.choice([0.10, 0.15, 0.20])
            temp      = np.random.choice([0.1, 0.3])
        else:
            top_k     = self.best_params["top_k"]
            threshold = self.best_params["threshold"]
            temp      = self.best_params["temp"]

        return {
            "top_k"     : int(top_k),
            "threshold" : float(threshold),
            "temp"      : float(temp),
        }


    def update(
        self,
        action      : dict,
        reward      : float,
        next_reward : float = 0.0
    ):

        self.step += 1

        top_k_key = f"top_k_{action['top_k']}"

        if   action["threshold"] <= 0.10 : thresh_key = "threshold_low"
        elif action["threshold"] <= 0.15 : thresh_key = "threshold_mid"
        else                             : thresh_key = "threshold_high"

        temp_key = (
            "temp_low" if action["temp"] < 0.2
            else "temp_high"
        )

        for key in [top_k_key, thresh_key, temp_key]:
            if key in self.q_values:
                old_q = self.q_values[key]
                new_q = old_q + self.lr * (
                    reward +
                    self.gamma * next_reward -
                    old_q
                )
                self.q_values[key] = round(new_q, 4)

        self.episode_rewards.append(reward)

        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if recent_avg > np.mean(
                self.episode_rewards[:-10] or [0]
            ):
                self.best_params = action.copy()

        self.epsilon = max(0.05, self.epsilon * 0.995)


    def get_best_params(self) -> dict:
        return self.best_params.copy()


    def get_stats(self) -> dict:

        if not self.episode_rewards:
            return {}

        return {
            "total_episodes" : len(self.episode_rewards),
            "avg_reward"     : round(float(np.mean(self.episode_rewards)), 4),
            "best_reward"    : round(float(max(self.episode_rewards)),     4),
            "final_epsilon"  : round(self.epsilon, 4),
            "best_params"    : self.best_params,
            "q_values"       : self.q_values,
        }