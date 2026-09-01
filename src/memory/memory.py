# src/memory/memory.py
# ============================================================
# Memory Layer for Doctor-Patient Oncology AI Simulation
# Tracks:
#   - Active cancer context across turns (e.g. "lung cancer")
#   - Pronoun & reference resolution ("this", "it", "that" -> cancer type)
#   - Turn history, question types, and clinical actions
# ============================================================

import os
import sys
import re
import json
from datetime import datetime


# ── Canonical Cancer Types ────────────────────────────────────
KNOWN_CANCERS = [
    "lung cancer", "breast cancer", "colon cancer",
    "colorectal cancer", "prostate cancer", "skin cancer",
    "melanoma", "leukemia", "lymphoma", "ovarian cancer",
    "cervical cancer", "pancreatic cancer", "liver cancer",
    "stomach cancer", "kidney cancer", "bladder cancer",
    "thyroid cancer", "brain tumor", "bone cancer",
    "esophageal cancer", "testicular cancer", "uterine cancer",
    "head and neck cancer", "oral cancer", "multiple myeloma"
]


class ConversationMemory:
    """
    Session-level memory manager that maintains dialogue history,
    active oncological topic, and resolves anaphoric references.
    """

    def __init__(self, session_id: str):
        self.session_id      = session_id
        self.history         = []
        self.cancer_context  = ""
        self.last_question   = ""
        self.last_answer     = ""
        self.turn_count      = 0

    def extract_cancer_from_text(self, text: str) -> str:
        """Extracts any known cancer type from a text string."""
        text_lower = text.lower()
        for cancer in KNOWN_CANCERS:
            if cancer in text_lower:
                return cancer
        return ""

    def add_turn(
        self,
        question      : str,
        answer        : str,
        cancer_type   : str = "",
        question_type : str = "general",
        action        : str = "answer",
        role          : str = "patient"
    ):
        """Records a new conversation turn and updates active cancer memory."""
        # Detect cancer if not explicitly provided
        detected_cancer = self.extract_cancer_from_text(question) or cancer_type

        if detected_cancer and detected_cancer not in ["cancer", "n/a", ""]:
            self.cancer_context = detected_cancer.lower()

        turn_entry = {
            "turn"          : self.turn_count + 1,
            "role"          : role,
            "question"      : question,
            "answer"        : answer,
            "cancer_type"   : self.cancer_context,
            "question_type" : question_type,
            "action"        : action,
            "timestamp"     : datetime.now().isoformat()
        }

        self.history.append(turn_entry)
        self.last_question = question
        self.last_answer   = answer
        self.turn_count   += 1

    def resolve_question(self, question: str) -> tuple:
        """
        Resolves references like 'this', 'it', 'that', 'this cancer'
        to the active cancer context stored in memory.

        Returns:
            (resolved_question: str, was_resolved: bool)
        """
        if not self.cancer_context:
            return question, False

        q_lower = question.lower().strip()

        # Specific reference phrases ordered from longest to shortest
        ref_patterns = [
            (r"\bthis cancer\b",    self.cancer_context),
            (r"\bthe cancer\b",     self.cancer_context),
            (r"\bthis disease\b",   self.cancer_context),
            (r"\bthis condition\b", self.cancer_context),
            (r"\bthis type\b",      self.cancer_context),
            (r"\bthis\b",           self.cancer_context),
            (r"\bit\b",             self.cancer_context),
            (r"\bthat\b",           self.cancer_context),
        ]

        resolved = question
        resolved_flag = False

        for pattern, rep in ref_patterns:
            if re.search(pattern, resolved, re.IGNORECASE):
                # Replace only the first occurrence
                resolved = re.sub(pattern, rep, resolved, count=1, flags=re.IGNORECASE)
                resolved_flag = True
                break

        return resolved, resolved_flag

    def get_context_str(self, last_n: int = 4) -> str:
        """Formats recent history into a clean dialogue summary."""
        if not self.history:
            return ""

        recent = self.history[-last_n:]
        lines = []
        for t in recent:
            lines.append(f"Patient: {t['question']}")
            lines.append(f"Doctor: {t['answer']}")

        return "\n".join(lines)

    def get_cancer_context(self) -> str:
        return self.cancer_context

    def clear(self):
        self.history        = []
        self.cancer_context = ""
        self.last_question  = ""
        self.last_answer    = ""
        self.turn_count     = 0


class MemoryManager:
    """Multi-session manager for conversation memory."""

    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id)
        return self.sessions[session_id]

    def resolve(self, session_id: str, question: str) -> tuple:
        session = self.get_session(session_id)
        return session.resolve_question(question)

    def add_turn(
        self,
        session_id    : str,
        question      : str,
        answer        : str,
        cancer_type   : str = "",
        question_type : str = "general",
        action        : str = "answer",
        role          : str = "patient"
    ):
        session = self.get_session(session_id)
        session.add_turn(
            question      = question,
            answer        = answer,
            cancer_type   = cancer_type,
            question_type = question_type,
            action        = action,
            role          = role
        )

    def get_cancer_context(self, session_id: str) -> str:
        return self.get_session(session_id).get_cancer_context()

    def get_context_str(self, session_id: str, last_n: int = 4) -> str:
        return self.get_session(session_id).get_context_str(last_n)

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].clear()
