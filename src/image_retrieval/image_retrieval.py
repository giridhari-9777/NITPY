# src/image_retrieval/image_retrieval.py

import os
import sys
import pickle
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# CONFIG
# ==================================================

IMAGE_EMBEDDINGS_PKL = "outputs/image_embeddings/image_embeddings.pkl"
PROCESSED_IMAGES     = "outputs/processed_images"
TOP_K                = 5


# ==================================================
# IMAGE RETRIEVAL CLASS
# ==================================================

class ImageRetrieval:

    def __init__(self):

        print("\nInitializing Image Retrieval...")

        # ── Load CLIP ────────────────────────────────
        print("  Loading CLIP model...")
        from transformers import CLIPProcessor, CLIPModel

        self.clip_model     = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model.eval()
        print("  CLIP loaded ✅")

        # ── Load text embedding model ─────────────────
        print("  Loading text model...")
        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print("  Text model loaded ✅")

        # ── Load image embeddings ─────────────────────
        print("  Loading image embeddings...")
        self.embeddings, self.image_paths = (
            self._load_embeddings()
        )

        # ── Detect embedding dim ──────────────────────
        if len(self.embeddings) > 0:
            self.img_dim = self.embeddings.shape[1]
            print(
                f"  Image dim    : {self.img_dim}"
            )
        else:
            self.img_dim = 768
            print("  No embeddings loaded!")

        print("\nImage Retrieval ready!\n")


    # ==================================================
    # LOAD EMBEDDINGS
    # ==================================================

    def _load_embeddings(self) -> tuple:

        if not os.path.exists(IMAGE_EMBEDDINGS_PKL):
            print(
                f"  No embeddings at "
                f"{IMAGE_EMBEDDINGS_PKL}"
            )
            print(
                "  Run: python3 src/image_embeddings/"
                "image_embedding_pipeline.py"
            )
            return np.array([]), []

        with open(IMAGE_EMBEDDINGS_PKL, "rb") as f:
            data = pickle.load(f)

        embeddings  = data["embeddings"]
        image_paths = data["paths"]

        print(
            f"  Loaded {len(image_paths)} embeddings"
        )
        print(f"  Shape : {embeddings.shape}")

        return embeddings, image_paths


    # ==================================================
    # GET TEXT EMBEDDING — SAME DIM AS IMAGE
    # ==================================================

    def _get_text_embedding(self, text: str) -> np.ndarray:

        inputs = self.clip_processor(
            text           = [text],
            return_tensors = "pt",
            padding        = True
        )

        with torch.no_grad():
            text_output = self.clip_model.text_model(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"]
            )
            embedding = text_output.pooler_output

        emb = embedding.cpu().numpy()[0]

        # Match image embedding dimension
        txt_dim = emb.shape[0]

        if txt_dim != self.img_dim:
            if txt_dim < self.img_dim:
                emb = np.pad(
                    emb,
                    (0, self.img_dim - txt_dim),
                    mode = "constant"
                )
            else:
                emb = emb[:self.img_dim]

        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb


    # ==================================================
    # GET IMAGE EMBEDDING FROM FILE
    # ==================================================

    def _get_image_embedding(
        self,
        image_path : str
    ) -> np.ndarray:

        from PIL import Image

        img = Image.open(image_path).convert("RGB")

        inputs = self.clip_processor(
            images         = img,
            return_tensors = "pt"
        )

        with torch.no_grad():
            vision_output = self.clip_model.vision_model(
                pixel_values = inputs["pixel_values"]
            )
            embedding = vision_output.pooler_output

        emb = embedding.cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb


    # ==================================================
    # SEARCH BY TEXT QUERY
    # ==================================================

    def search_by_text(
        self,
        query : str,
        top_k : int = TOP_K
    ) -> list:

        if len(self.embeddings) == 0:
            print("  No embeddings available!")
            return []

        # Get text embedding
        text_emb = self._get_text_embedding(query)

        # Normalize image embeddings
        norms           = np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        norm_embeddings = self.embeddings / (norms + 1e-8)

        # Cosine similarity
        scores  = np.dot(norm_embeddings, text_emb)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx):
            results.append({
                "rank"        : rank + 1,
                "image_path"  : self.image_paths[idx],
                "image_name"  : os.path.basename(
                    self.image_paths[idx]
                ),
                "score"       : round(float(scores[idx]), 4),
                "source_pdf"  : self._get_source_pdf(
                    self.image_paths[idx]
                ),
                "page_number" : self._get_page_number(
                    self.image_paths[idx]
                )
            })

        return results


    # ==================================================
    # SEARCH BY IMAGE
    # ==================================================

    def search_by_image(
        self,
        image_path : str,
        top_k      : int = TOP_K
    ) -> list:

        if len(self.embeddings) == 0:
            print("  No embeddings available!")
            return []

        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            return []

        # Get image embedding
        query_emb = self._get_image_embedding(image_path)

        # Normalize
        norms           = np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        norm_embeddings = self.embeddings / (norms + 1e-8)

        # Cosine similarity
        scores  = np.dot(norm_embeddings, query_emb)
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Skip the query image itself
        results = []
        for rank, idx in enumerate(top_idx):

            if self.image_paths[idx] == image_path:
                continue

            results.append({
                "rank"        : rank + 1,
                "image_path"  : self.image_paths[idx],
                "image_name"  : os.path.basename(
                    self.image_paths[idx]
                ),
                "score"       : round(float(scores[idx]), 4),
                "source_pdf"  : self._get_source_pdf(
                    self.image_paths[idx]
                ),
                "page_number" : self._get_page_number(
                    self.image_paths[idx]
                )
            })

            if len(results) >= top_k:
                break

        return results


    # ==================================================
    # MEDICAL IMAGE SEARCH
    # (Combines text + medical context)
    # ==================================================

    def medical_search(
        self,
        query        : str,
        cancer_type  : str = "",
        question_type: str = "",
        top_k        : int = TOP_K
    ) -> list:

        # Build enhanced medical query
        medical_query = query

        if cancer_type and cancer_type != "cancer":
            medical_query = (
                f"{query} {cancer_type} medical imaging"
            )

        if question_type:
            type_terms = {
                "symptoms"   : "clinical presentation",
                "diagnosis"  : "diagnostic imaging scan",
                "treatment"  : "treatment procedure",
                "prognosis"  : "outcome survival",
                "radiology"  : "radiological imaging",
                "pathology"  : "pathology histology",
                "staging"    : "tumor staging",
                "prevention" : "screening prevention"
            }
            term = type_terms.get(question_type, "")
            if term:
                medical_query = f"{medical_query} {term}"

        print(f"\n  Medical query: {medical_query}")

        return self.search_by_text(
            query = medical_query,
            top_k = top_k
        )


    # ==================================================
    # HELPER — GET SOURCE PDF
    # ==================================================

    def _get_source_pdf(self, image_path: str) -> str:

        name = os.path.basename(image_path)

        # Format: pdfname_pageN.png
        if "_page" in name:
            return name.split("_page")[0] + ".pdf"

        return "unknown"


    # ==================================================
    # HELPER — GET PAGE NUMBER
    # ==================================================

    def _get_page_number(self, image_path: str) -> int:

        name = os.path.basename(image_path)

        try:
            if "_page" in name:
                page_part = name.split("_page")[1]
                page_num  = page_part.split(".")[0]
                return int(page_num)
        except Exception:
            pass

        return 0


    # ==================================================
    # FORMAT RESULTS
    # ==================================================

    def format_results(self, results: list) -> str:

        if not results:
            return "No relevant images found."

        lines = []

        for r in results:
            lines.append(
                f"\n[{r['rank']}] {r['image_name']}"
                f"\n     Source : {r['source_pdf']}"
                f"\n     Page   : {r['page_number']}"
                f"\n     Score  : {r['score']}"
            )

        return "\n".join(lines)


    # ==================================================
    # PRINT RESULTS
    # ==================================================

    def print_results(
        self,
        results : list,
        query   : str = ""
    ):

        print(f"\n{'='*60}")
        if query:
            print(f"  Query  : {query}")
        print(f"  Results: {len(results)}")
        print(f"{'='*60}")

        if not results:
            print("  No relevant images found.")
            return

        for r in results:
            print(f"\n  [{r['rank']}] {r['image_name']}")
            print(f"       Source : {r['source_pdf']}")
            print(f"       Page   : {r['page_number']}")
            print(f"       Score  : {r['score']}")

        print(f"{'='*60}\n")


    # ==================================================
    # STATS
    # ==================================================

    def stats(self):

        print(f"\n{'='*60}")
        print(f"  IMAGE RETRIEVAL STATS")
        print(f"{'='*60}")
        print(
            f"  Total Images     : "
            f"{len(self.image_paths)}"
        )
        print(
            f"  Embedding Shape  : "
            f"{self.embeddings.shape}"
            if len(self.embeddings) > 0
            else "  Embedding Shape  : None"
        )
        print(f"  Image Dim        : {self.img_dim}")
        print(f"{'='*60}\n")


# ==================================================
# MAIN — Terminal Test
# ==================================================

if __name__ == "__main__":

    retrieval = ImageRetrieval()
    retrieval.stats()

    # Test queries
    test_queries = [
        {
            "query"         : "lung cancer xray chest",
            "cancer_type"   : "lung cancer",
            "question_type" : "radiology"
        },
        {
            "query"         : "breast cancer mammogram scan",
            "cancer_type"   : "breast cancer",
            "question_type" : "diagnosis"
        },
        {
            "query"         : "cancer cell pathology slide",
            "cancer_type"   : "cancer",
            "question_type" : "pathology"
        },
        {
            "query"         : "tumor staging classification",
            "cancer_type"   : "cancer",
            "question_type" : "staging"
        },
        {
            "query"         : "chemotherapy treatment procedure",
            "cancer_type"   : "cancer",
            "question_type" : "treatment"
        },
    ]

    for item in test_queries:

        print(f"\n{'─'*60}")
        print(f"Query        : {item['query']}")
        print(f"Cancer Type  : {item['cancer_type']}")
        print(f"Question Type: {item['question_type']}")

        results = retrieval.medical_search(
            query         = item["query"],
            cancer_type   = item["cancer_type"],
            question_type = item["question_type"],
            top_k         = 3
        )

        retrieval.print_results(results, item["query"])

    # Interactive terminal search
    print("\n" + "="*60)
    print("  INTERACTIVE IMAGE SEARCH")
    print("  Type query to search medical images")
    print("  Type 'exit' to quit")
    print("="*60 + "\n")

    while True:

        try:
            query = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        results = retrieval.search_by_text(
            query = query,
            top_k = 5
        )

        retrieval.print_results(results, query)