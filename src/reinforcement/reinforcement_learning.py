# src/reinforcement/reinforcement_learning.py

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
# CONFIG
# ==================================================

FEEDBACK_FILE    = "outputs/feedback/feedback_log.json"
RL_MODEL_FILE    = "outputs/feedback/rl_model.json"
REWARD_THRESHOLD = 3.5    # Min score to consider good
LEARNING_RATE    = 0.01


# ==================================================
# REWARD CALCULATOR
# ==================================================

class RewardCalculator:

    def __init__(self):
        pass


    # ==================================================
    # CALCULATE REWARD FROM FEEDBACK
    # ==================================================

    def calculate_reward(
        self,
        feedback      : dict,
        hallucination : dict = {}
    ) -> dict:

        # ── User rating reward (0-5) ─────────────────
        user_rating   = feedback.get("user_rating", 3)
        rating_reward = (user_rating - 3) * 0.4

        # ── Hallucination reward ─────────────────────
        hall_score    = hallucination.get("score", 3.0)
        hall_reward   = (hall_score - 3.0) * 0.3

        # ── Answer length reward ─────────────────────
        answer        = feedback.get("answer", "")
        word_count    = len(answer.split())

        if   50  <= word_count <= 250 : length_reward =  0.1
        elif 30  <= word_count <  50  : length_reward =  0.05
        elif word_count < 30          : length_reward = -0.2
        else                          : length_reward =  0.0

        # ── Source quality reward ────────────────────
        sources       = feedback.get("sources", [])
        num_sources   = len(sources)

        if   num_sources >= 3 : source_reward =  0.1
        elif num_sources == 2 : source_reward =  0.05
        elif num_sources == 1 : source_reward =  0.0
        else                  : source_reward = -0.1

        # ── Safety reward ────────────────────────────
        safety       = hallucination.get("safety", "LOW")
        safety_reward = (
             0.1 if safety == "LOW"    else
            -0.1 if safety == "HIGH"   else
             0.0
        )

        # ── Total reward ─────────────────────────────
        total_reward = (
            rating_reward  +
            hall_reward    +
            length_reward  +
            source_reward  +
            safety_reward
        )

        total_reward = round(
            max(-1.0, min(1.0, total_reward)), 4
        )

        return {
            "total_reward"  : total_reward,
            "rating_reward" : round(rating_reward,  4),
            "hall_reward"   : round(hall_reward,     4),
            "length_reward" : round(length_reward,   4),
            "source_reward" : round(source_reward,   4),
            "safety_reward" : round(safety_reward,   4),
            "is_positive"   : total_reward > 0
        }


# ==================================================
# FEEDBACK LOGGER
# ==================================================

class FeedbackLogger:

    def __init__(self):

        os.makedirs(
            os.path.dirname(FEEDBACK_FILE),
            exist_ok=True
        )

        self.feedback_log = []
        self._load()


    def _load(self):

        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r") as f:
                    self.feedback_log = json.load(f)
                print(
                    f"  Loaded {len(self.feedback_log)}"
                    f" feedback records"
                )
            except Exception:
                self.feedback_log = []


    def save(self):

        os.makedirs(
            os.path.dirname(FEEDBACK_FILE),
            exist_ok=True
        )

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(
                self.feedback_log,
                f,
                indent=2
            )


    def log(
        self,
        question      : str,
        answer        : str,
        user_rating   : float,
        question_type : str,
        cancer_type   : str,
        sources       : list,
        hallucination : dict,
        reward        : dict,
        session_id    : str = "default"
    ):

        record = {
            "id"            : len(self.feedback_log) + 1,
            "timestamp"     : datetime.now().isoformat(),
            "session_id"    : session_id,
            "question"      : question,
            "answer"        : answer[:500],
            "user_rating"   : user_rating,
            "question_type" : question_type,
            "cancer_type"   : cancer_type,
            "sources"       : sources,
            "hallucination" : hallucination,
            "reward"        : reward
        }

        self.feedback_log.append(record)
        self.save()

        return record


    def get_all(self) -> list:
        return self.feedback_log


    def get_by_type(
        self,
        question_type : str
    ) -> list:
        return [
            r for r in self.feedback_log
            if r["question_type"] == question_type
        ]


    def get_stats(self) -> dict:

        if not self.feedback_log:
            return {
                "total"        : 0,
                "avg_rating"   : 0.0,
                "avg_reward"   : 0.0,
                "positive_pct" : 0.0
            }

        ratings = [
            r["user_rating"]
            for r in self.feedback_log
        ]
        rewards = [
            r["reward"]["total_reward"]
            for r in self.feedback_log
        ]
        positive = sum(
            1 for r in self.feedback_log
            if r["reward"]["is_positive"]
        )

        return {
            "total"        : len(self.feedback_log),
            "avg_rating"   : round(float(np.mean(ratings)), 2),
            "avg_reward"   : round(float(np.mean(rewards)), 4),
            "positive_pct" : round(
                positive / len(self.feedback_log) * 100, 1
            ),
            "by_type"      : self._stats_by_type()
        }


    def _stats_by_type(self) -> dict:

        by_type = {}

        for r in self.feedback_log:
            qt = r["question_type"]
            if qt not in by_type:
                by_type[qt] = {
                    "count"   : 0,
                    "ratings" : []
                }
            by_type[qt]["count"] += 1
            by_type[qt]["ratings"].append(
                r["user_rating"]
            )

        result = {}
        for qt, data in by_type.items():
            result[qt] = {
                "count"      : data["count"],
                "avg_rating" : round(
                    float(np.mean(data["ratings"])), 2
                )
            }

        return result


# ==================================================
# RL POLICY — Learns which retrieval params work best
# ==================================================

class RLPolicy:

    def __init__(self):

        # Policy parameters
        self.params = {
            "top_k"           : 5,
            "temperature"     : 0.3,
            "min_score"       : 0.35,
            "max_retries"     : 1,
            "context_length"  : 400,
        }

        # Q-values for each param adjustment
        self.q_values = {
            "increase_top_k"      : 0.0,
            "decrease_top_k"      : 0.0,
            "increase_temperature": 0.0,
            "decrease_temperature": 0.0,
            "increase_context"    : 0.0,
            "decrease_context"    : 0.0,
        }

        # Load saved policy
        self._load()


    def _load(self):

        if os.path.exists(RL_MODEL_FILE):
            try:
                with open(RL_MODEL_FILE, "r") as f:
                    data = json.load(f)
                self.params   = data.get("params",   self.params)
                self.q_values = data.get("q_values", self.q_values)
                print("  RL policy loaded ✅")
            except Exception:
                pass


    def save(self):

        os.makedirs(
            os.path.dirname(RL_MODEL_FILE),
            exist_ok=True
        )

        with open(RL_MODEL_FILE, "w") as f:
            json.dump(
                {
                    "params"   : self.params,
                    "q_values" : self.q_values,
                    "updated"  : datetime.now().isoformat()
                },
                f,
                indent=2
            )


    # ==================================================
    # UPDATE Q-VALUES FROM REWARD
    # ==================================================

    def update(
        self,
        action : str,
        reward : float
    ):

        if action not in self.q_values:
            return

        # Q-learning update
        old_q = self.q_values[action]
        new_q = old_q + LEARNING_RATE * (
            reward - old_q
        )
        self.q_values[action] = round(new_q, 4)

        self.save()


    # ==================================================
    # GET BEST ACTION
    # ==================================================

    def get_best_action(self) -> str:

        return max(
            self.q_values,
            key=lambda x: self.q_values[x]
        )


    # ==================================================
    # APPLY BEST ACTION TO PARAMS
    # ==================================================

    def apply_best_action(self) -> dict:

        action = self.get_best_action()

        if action == "increase_top_k":
            self.params["top_k"] = min(
                self.params["top_k"] + 1, 10
            )
        elif action == "decrease_top_k":
            self.params["top_k"] = max(
                self.params["top_k"] - 1, 3
            )
        elif action == "increase_temperature":
            self.params["temperature"] = min(
                self.params["temperature"] + 0.05, 0.8
            )
        elif action == "decrease_temperature":
            self.params["temperature"] = max(
                self.params["temperature"] - 0.05, 0.1
            )
        elif action == "increase_context":
            self.params["context_length"] = min(
                self.params["context_length"] + 100, 800
            )
        elif action == "decrease_context":
            self.params["context_length"] = max(
                self.params["context_length"] - 100, 200
            )

        self.save()

        return {
            "action" : action,
            "params" : self.params
        }


    def get_params(self) -> dict:
        return self.params.copy()


# ==================================================
# REINFORCEMENT LEARNING SYSTEM
# ==================================================

class ReinforcementLearning:

    def __init__(self):

        print("\nInitializing Reinforcement Learning...")

        self.reward_calc = RewardCalculator()
        self.logger      = FeedbackLogger()
        self.policy      = RLPolicy()

        print("RL System ready!\n")


    # ==================================================
    # COLLECT FEEDBACK
    # ==================================================

    def collect_feedback(
        self,
        question      : str,
        answer        : str,
        question_type : str,
        cancer_type   : str,
        sources       : list,
        hallucination : dict,
        session_id    : str = "default"
    ) -> dict:

        # Auto-calculate rating from hallucination score
        hall_score  = hallucination.get("score", 3.0)
        auto_rating = round(
            min(hall_score * 1.1, 5.0), 1
        )

        # Get terminal rating if interactive
        try:
            print(
                f"\n  Rate this answer (1-5) "
                f"[default: {auto_rating}]: ",
                end=""
            )

            import sys
            import select

            # Wait 3 seconds for input
            rlist, _, _ = select.select(
                [sys.stdin], [], [], 3
            )

            if rlist:
                rating_input = sys.stdin.readline().strip()
                user_rating  = float(rating_input) if rating_input else auto_rating
                user_rating  = max(1.0, min(5.0, user_rating))
            else:
                user_rating = auto_rating
                print(f"{auto_rating} (auto)")

        except Exception:
            user_rating = auto_rating

        # Calculate reward
        feedback = {
            "answer"      : answer,
            "user_rating" : user_rating,
            "sources"     : sources
        }

        reward = self.reward_calc.calculate_reward(
            feedback      = feedback,
            hallucination = hallucination
        )

        # Log feedback
        record = self.logger.log(
            question      = question,
            answer        = answer,
            user_rating   = user_rating,
            question_type = question_type,
            cancer_type   = cancer_type,
            sources       = sources,
            hallucination = hallucination,
            reward        = reward,
            session_id    = session_id
        )

        # Update RL policy
        self._update_policy(reward["total_reward"])

        return {
            "user_rating" : user_rating,
            "reward"      : reward,
            "record_id"   : record["id"]
        }


    # ==================================================
    # UPDATE POLICY FROM REWARD
    # ==================================================

    def _update_policy(self, total_reward: float):

        # Determine which actions led to this reward
        params = self.policy.get_params()

        # Update relevant Q-values
        if params["top_k"] > 5:
            self.policy.update(
                "increase_top_k", total_reward
            )
        else:
            self.policy.update(
                "decrease_top_k", total_reward
            )

        if params["temperature"] > 0.3:
            self.policy.update(
                "increase_temperature", total_reward
            )
        else:
            self.policy.update(
                "decrease_temperature", total_reward
            )

        # Apply best action periodically
        total_records = len(self.logger.feedback_log)

        if total_records % 10 == 0 and total_records > 0:
            result = self.policy.apply_best_action()
            print(
                f"\n  RL Update: {result['action']}"
                f" → params: {result['params']}"
            )


    # ==================================================
    # GET OPTIMIZED PARAMS
    # ==================================================

    def get_optimized_params(self) -> dict:
        return self.policy.get_params()


    # ==================================================
    # PRINT FEEDBACK STATS
    # ==================================================

    def print_stats(self):

        stats = self.logger.get_stats()

        print(f"\n{'='*60}")
        print(f"  REINFORCEMENT LEARNING STATS")
        print(f"{'='*60}")
        print(f"  Total Feedback   : {stats['total']}")
        print(f"  Avg Rating       : {stats['avg_rating']} / 5.0")
        print(f"  Avg Reward       : {stats['avg_reward']}")
        print(f"  Positive Pct     : {stats['positive_pct']}%")

        if stats.get("by_type"):
            print(f"\n  By Question Type:")
            for qt, data in stats["by_type"].items():
                print(
                    f"     {qt:<15} : "
                    f"count={data['count']} "
                    f"avg={data['avg_rating']}"
                )

        print(f"\n  RL Policy Params:")
        params = self.policy.get_params()
        for k, v in params.items():
            print(f"     {k:<20} : {v}")

        print(f"\n  Q-Values:")
        for action, q in self.policy.q_values.items():
            print(f"     {action:<30} : {q}")

        print(f"{'='*60}\n")


    # ==================================================
    # SIMULATE FEEDBACK (for testing)
    # ==================================================

    def simulate_feedback(
        self,
        n_episodes : int = 20
    ):

        print(f"\nSimulating {n_episodes} feedback episodes...")

        question_types = [
            "symptoms", "treatment", "diagnosis",
            "prognosis", "staging"
        ]

        cancer_types = [
            "lung cancer", "breast cancer",
            "colon cancer", "leukemia"
        ]

        for i in range(n_episodes):

            qt     = np.random.choice(question_types)
            ct     = np.random.choice(cancer_types)
            rating = np.random.uniform(3.0, 5.0)

            feedback = {
                "answer"      : (
                    "Based on medical literature, "
                    "the treatment may include "
                    "chemotherapy and surgery. "
                    "Please consult your oncologist."
                ),
                "user_rating" : rating,
                "sources"     : [
                    {"source": "basics_of_oncology"},
                    {"source": "MD_Anderson_Manual"},
                ]
            }

            hall = {
                "score"  : round(np.random.uniform(3.5, 4.8), 2),
                "safety" : "LOW"
            }

            reward = self.reward_calc.calculate_reward(
                feedback      = feedback,
                hallucination = hall
            )

            self.logger.log(
                question      = f"Test question {i+1} about {ct}?",
                answer        = feedback["answer"],
                user_rating   = rating,
                question_type = qt,
                cancer_type   = ct,
                sources       = feedback["sources"],
                hallucination = hall,
                reward        = reward,
                session_id    = "simulation"
            )

            self._update_policy(reward["total_reward"])

            if (i + 1) % 5 == 0:
                print(f"  Episode {i+1}/{n_episodes} ✅")

        print("\nSimulation complete!")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    rl = ReinforcementLearning()

    print("\nOptions:")
    print("1. Run simulation")
    print("2. Show stats")
    print("3. Both")

    try:
        choice = input("\nChoice (1/2/3): ").strip()
    except Exception:
        choice = "3"

    if choice == "1":
        rl.simulate_feedback(n_episodes=20)

    elif choice == "2":
        rl.print_stats()

    elif choice == "3":
        rl.simulate_feedback(n_episodes=20)
        rl.print_stats()

    else:
        rl.simulate_feedback(n_episodes=20)
        rl.print_stats()