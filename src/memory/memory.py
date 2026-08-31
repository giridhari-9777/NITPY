# src/memory/memory.py

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# MEMORY CLASS
# ==================================================

class ConversationMemory:

    def __init__(self, session_id: str = "default"):

        self.session_id    = session_id
        self.history       = []   # Full conversation
        self.cancer_context = ""  # Last cancer type discussed
        self.last_question  = ""  # Last patient question
        self.last_answer    = ""  # Last doctor answer
        self.turn_count     = 0

        print(f"  Memory initialized for: {session_id}")


    # ==================================================
    # ADD TURN
    # ==================================================

    def add_turn(
        self,
        question      : str,
        answer        : str,
        cancer_type   : str = "",
        question_type : str = ""
    ):

        turn = {
            "turn"          : self.turn_count + 1,
            "timestamp"     : datetime.now().isoformat(),
            "question"      : question,
            "answer"        : answer,
            "cancer_type"   : cancer_type,
            "question_type" : question_type
        }

        self.history.append(turn)

        # Update context
        if cancer_type and cancer_type != "cancer":
            self.cancer_context = cancer_type

        self.last_question = question
        self.last_answer   = answer
        self.turn_count   += 1


    # ==================================================
    # RESOLVE REFERENCES
    # "this", "it", "that" → actual cancer type
    # ==================================================

    def resolve_question(self, question: str) -> str:

        question_lower = question.lower()

        # Reference words that mean previous cancer
        ref_words = [
            "this", "it", "that", "this cancer",
            "this disease", "the cancer", "the disease",
            "this condition", "this type"
        ]

        has_reference = any(
            ref in question_lower
            for ref in ref_words
        )

        # If reference found and we have context
        if has_reference and self.cancer_context:
            resolved = question

            for ref in ref_words:
                if ref in question_lower:
                    resolved = resolved.replace(
                        ref, self.cancer_context
                    ).replace(
                        ref.capitalize(),
                        self.cancer_context.capitalize()
                    )

            if resolved != question:
                print(
                    f"  Memory resolved: "
                    f"'{question}' → '{resolved}'"
                )

            return resolved

        return question


    # ==================================================
    # GET CONTEXT FOR PROMPT
    # ==================================================

    def get_context(self, last_n: int = 4) -> str:

        if not self.history:
            return ""

        recent = self.history[-last_n:]
        lines  = []

        for turn in recent:
            lines.append(
                f"Patient: {turn['question']}"
            )
            # Only first 150 chars of answer
            short_answer = turn["answer"][:150]
            if len(turn["answer"]) > 150:
                short_answer += "..."
            lines.append(
                f"Doctor: {short_answer}"
            )

        return "\n".join(lines)


    # ==================================================
    # GET CANCER CONTEXT
    # ==================================================

    def get_cancer_context(self) -> str:
        return self.cancer_context


    # ==================================================
    # CLEAR MEMORY
    # ==================================================

    def clear(self):
        self.history        = []
        self.cancer_context = ""
        self.last_question  = ""
        self.last_answer    = ""
        self.turn_count     = 0
        print(f"  Memory cleared for: {self.session_id}")


    # ==================================================
    # PRINT HISTORY
    # ==================================================

    def print_history(self):

        print(f"\n{'='*60}")
        print(f"  CONVERSATION MEMORY")
        print(f"  Session : {self.session_id}")
        print(f"  Turns   : {self.turn_count}")
        print(f"  Cancer  : {self.cancer_context}")
        print(f"{'='*60}")

        for turn in self.history:
            print(f"\n  [{turn['turn']}] Patient: {turn['question']}")
            print(f"       Doctor : {turn['answer'][:100]}...")

        print(f"{'='*60}\n")


# ==================================================
# MEMORY MANAGER — Manages multiple sessions
# ==================================================

class MemoryManager:

    def __init__(self):

        self.sessions    = {}
        self.memory_file = "outputs/memory/sessions.json"

        os.makedirs("outputs/memory", exist_ok=True)
        self._load()

        print("  Memory Manager ready ✅")


    def _load(self):

        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)

                for sid, turns in data.items():
                    mem = ConversationMemory(sid)
                    mem.history    = turns.get("history", [])
                    mem.turn_count = len(mem.history)

                    # Restore cancer context
                    for turn in reversed(mem.history):
                        if turn.get("cancer_type") and \
                           turn["cancer_type"] != "cancer":
                            mem.cancer_context = turn["cancer_type"]
                            break

                    self.sessions[sid] = mem

                print(
                    f"  Loaded {len(self.sessions)}"
                    f" sessions from memory"
                )

            except Exception as e:
                print(f"  Memory load error: {e}")


    def _save(self):

        data = {}
        for sid, mem in self.sessions.items():
            data[sid] = {
                "history"        : mem.history,
                "cancer_context" : mem.cancer_context
            }

        try:
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  Memory save error: {e}")


    def get_session(
        self,
        session_id : str
    ) -> ConversationMemory:

        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(
                session_id
            )

        return self.sessions[session_id]


    def add_turn(
        self,
        session_id    : str,
        question      : str,
        answer        : str,
        cancer_type   : str = "",
        question_type : str = ""
    ):

        mem = self.get_session(session_id)

        mem.add_turn(
            question      = question,
            answer        = answer,
            cancer_type   = cancer_type,
            question_type = question_type
        )

        self._save()


    def resolve_question(
        self,
        session_id : str,
        question   : str
    ) -> str:

        mem = self.get_session(session_id)
        return mem.resolve_question(question)


    def get_context(
        self,
        session_id : str,
        last_n     : int = 4
    ) -> str:

        mem = self.get_session(session_id)
        return mem.get_context(last_n)


    def get_cancer_context(
        self,
        session_id : str
    ) -> str:

        mem = self.get_session(session_id)
        return mem.get_cancer_context()


    def clear_session(self, session_id: str):

        if session_id in self.sessions:
            self.sessions[session_id].clear()
            self._save()


# ==================================================
# MAIN — Test
# ==================================================

if __name__ == "__main__":

    manager = MemoryManager()
    session = "test_patient_001"

    print("\nSimulating doctor-patient conversation...\n")

    # Turn 1
    manager.add_turn(
        session_id    = session,
        question      = "What is breast cancer?",
        answer        = "Breast cancer is a malignant tumor that develops in breast tissue. It may present as a lump, skin changes, or nipple discharge.",
        cancer_type   = "breast cancer",
        question_type = "general"
    )

    # Turn 2 — with reference "this"
    q2 = "What is the treatment for this?"
    resolved_q2 = manager.resolve_question(session, q2)

    print(f"Original  : {q2}")
    print(f"Resolved  : {resolved_q2}")

    manager.add_turn(
        session_id    = session,
        question      = resolved_q2,
        answer        = "Treatment for breast cancer includes surgery, chemotherapy, radiation and hormone therapy.",
        cancer_type   = "breast cancer",
        question_type = "treatment"
    )

    # Turn 3 — with reference "it"
    q3 = "What are the survival rates for it?"
    resolved_q3 = manager.resolve_question(session, q3)

    print(f"\nOriginal  : {q3}")
    print(f"Resolved  : {resolved_q3}")

    # Show context
    print(f"\nConversation Context:")
    print(manager.get_context(session))

    # Print history
    mem = manager.get_session(session)
    mem.print_history()