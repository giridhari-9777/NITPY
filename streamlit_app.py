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

# ── Extract ChromaDB from archive parts if not present ────────
def ensure_chroma():
    chroma_path = os.path.join(ROOT, "chroma_db")
    sqlite_path = os.path.join(chroma_path, "chroma.sqlite3")
    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 10 * 1024 * 1024:
        return
    archive_dir = os.path.join(ROOT, "chroma_db_archive")
    if not os.path.exists(archive_dir):
        return
    import glob, tarfile, io
    part_files = sorted(glob.glob(os.path.join(archive_dir, "chroma_db.tar.gz.part_*")))
    if part_files:
        print(f"Extracting ChromaDB ({len(part_files)} parts) into {ROOT}...")
        try:
            combined = bytearray()
            for p in part_files:
                with open(p, "rb") as f:
                    combined.extend(f.read())
            bio = io.BytesIO(combined)
            with tarfile.open(fileobj=bio, mode="r:gz") as tar:
                tar.extractall(path=ROOT)
            print("ChromaDB extracted successfully!")
        except Exception as e:
            print(f"ChromaDB extraction error: {e}")

ensure_chroma()

# ── Run UI ────────────────────────────────────────────────────
try:
    from src.ui import main
except ImportError:
    from ui import main

if __name__ == "__main__":
    main()
