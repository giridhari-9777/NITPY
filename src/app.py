# streamlit_app.py
# ============================================================
# Render Cloud Entry Point
# Place this file in ROOT of project (same level as src/)
# ============================================================

import os
import sys

# ── Fix paths so 'agents' module is always found ─────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

# ── Run UI ────────────────────────────────────────────────────
from ui import main

if __name__ == "__main__":
    main()
