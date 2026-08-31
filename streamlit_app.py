# streamlit_app.py
# ============================================================
# Render Cloud Entry Point
# Place this file in ROOT of project (same level as src/)
# ============================================================

import os
import sys

# ── Fix paths so 'agents' and 'src' modules are always found ──
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ── Run UI ────────────────────────────────────────────────────
try:
    from src.ui import main
except ImportError:
    from ui import main

if __name__ == "__main__":
    main()
