# src/image_embeddings/image_embedding_pipeline.py

import os
import sys
import pickle
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# CONFIG
# ==================================================

IMAGE_FOLDER  = "outputs/extracted_images"
OUTPUT_FOLDER = "outputs/image_embeddings"
OUTPUT_PKL    = "outputs/image_embeddings/image_embeddings.pkl"
SUPPORTED_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


# ==================================================
# CREATE FOLDERS
# ==================================================

def create_folders():
    for folder in [
        "outputs",
        "outputs/extracted_images",
        "outputs/image_embeddings"
    ]:
        os.makedirs(folder, exist_ok=True)
    print("  Folders ready ✅")


# ==================================================
# SAFE IMAGE LOADER
# ==================================================

def safe_load_image(image_path: str):

    from PIL import Image
    import io

    try:
        img = Image.open(image_path)
        img.load()
        return img.convert("RGB")
    except Exception:
        pass

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        pass

    try:
        import cv2
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            from PIL import Image
            return Image.fromarray(img_cv)
    except Exception:
        pass

    return None


# ==================================================
# EXTRACT IMAGES FROM PDFs
# ==================================================

def extract_images_from_pdfs(
    data_folder : str = "data",
    output_dir  : str = "outputs/extracted_images"
):

    try:
        import fitz

        os.makedirs(output_dir, exist_ok=True)

        pdf_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.lower().endswith(".pdf")
        ])

        if not pdf_files:
            print(f"  No PDFs in {data_folder}")
            return

        print(f"\nExtracting from {len(pdf_files)} PDFs...")
        print("="*60)

        total = 0

        for i, pdf_path in enumerate(pdf_files):

            pdf_name = os.path.splitext(
                os.path.basename(pdf_path)
            )[0]

            print(
                f"[{i+1}/{len(pdf_files)}] "
                f"{os.path.basename(pdf_path)}"
            )

            try:
                doc       = fitz.open(pdf_path)
                img_count = 0

                for page_num in range(len(doc)):

                    page = doc[page_num]

                    try:
                        mat  = fitz.Matrix(1.5, 1.5)
                        pix  = page.get_pixmap(matrix=mat)

                        img_filename = (
                            f"{pdf_name}_page{page_num+1}.png"
                        )
                        img_path = os.path.join(
                            output_dir, img_filename
                        )

                        pix.save(img_path)

                        test = safe_load_image(img_path)
                        if test is not None:
                            img_count += 1
                        else:
                            os.remove(img_path)

                    except Exception:
                        continue

                doc.close()
                total += img_count
                print(f"  → {img_count} page images")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        print(f"\nTotal extracted: {total} ✅")

    except ImportError:
        print("Run: pip install PyMuPDF")


# ==================================================
# LOAD CLIP MODEL
# ==================================================

def load_clip_model():

    print("\nLoading CLIP Model...")

    model     = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model.eval()

    # ── Detect correct embedding dim ─────────────
    from PIL import Image

    test_img    = Image.new("RGB", (224, 224), "white")
    test_inputs = processor(
        images         = test_img,
        return_tensors = "pt"
    )

    with torch.no_grad():
        test_out = model.vision_model(
            pixel_values = test_inputs["pixel_values"]
        )
        img_dim = test_out.pooler_output.shape[-1]

    with torch.no_grad():
        test_text = processor(
            text           = ["test"],
            return_tensors = "pt",
            padding        = True
        )
        test_text_out = model.text_model(
            input_ids      = test_text["input_ids"],
            attention_mask = test_text["attention_mask"]
        )
        txt_dim = test_text_out.pooler_output.shape[-1]

    print(f"  Image dim : {img_dim}")
    print(f"  Text dim  : {txt_dim}")
    print("CLIP Model loaded ✅\n")

    return model, processor, img_dim, txt_dim


# ==================================================
# GET IMAGE EMBEDDING
# ==================================================

def get_image_embedding(
    image,
    model,
    processor
) -> np.ndarray:

    inputs = processor(
        images         = image,
        return_tensors = "pt"
    )

    with torch.no_grad():
        vision_output = model.vision_model(
            pixel_values = inputs["pixel_values"]
        )
        embedding = vision_output.pooler_output

    return embedding.cpu().numpy()[0]


# ==================================================
# GET TEXT EMBEDDING — SAME DIM AS IMAGE
# ==================================================

def get_text_embedding(
    text      : str,
    model,
    processor
) -> np.ndarray:

    inputs = processor(
        text           = [text],
        return_tensors = "pt",
        padding        = True
    )

    with torch.no_grad():
        text_output = model.text_model(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"]
        )
        embedding = text_output.pooler_output

    return embedding.cpu().numpy()[0]


# ==================================================
# GENERATE IMAGE EMBEDDINGS
# ==================================================

def generate_image_embeddings(
    image_folder : str = IMAGE_FOLDER
) -> tuple:

    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        extract_images_from_pdfs()

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not image_files:
        print(f"No images in {image_folder}")
        return np.array([]), []

    print(f"\nFound {len(image_files)} images")
    print("="*60)

    model, processor, img_dim, txt_dim = load_clip_model()

    print(f"  Using image embedding dim: {img_dim}")

    image_embeddings = []
    image_paths      = []
    failed           = 0
    success          = 0

    for i, image_file in enumerate(image_files):

        image_path = os.path.join(image_folder, image_file)

        try:
            image = safe_load_image(image_path)

            if image is None:
                failed += 1
                continue

            if image.width < 32 or image.height < 32:
                failed += 1
                continue

            embedding = get_image_embedding(
                image     = image,
                model     = model,
                processor = processor
            )

            image_embeddings.append(embedding)
            image_paths.append(image_path)
            success += 1

            if success % 200 == 0:
                print(
                    f"  Embedded {success}/{len(image_files)}"
                    f" (failed: {failed})"
                )

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Error: {image_file} → {e}")
            continue

    print(f"\n  Success : {success}")
    print(f"  Failed  : {failed}")

    if not image_embeddings:
        return np.array([]), []

    embeddings_array = np.array(
        image_embeddings,
        dtype = np.float32
    )

    print(f"  Shape   : {embeddings_array.shape}")

    return embeddings_array, image_paths


# ==================================================
# SAVE EMBEDDINGS
# ==================================================

def save_embeddings(
    embeddings  : np.ndarray,
    image_paths : list,
    output_path : str = OUTPUT_PKL
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok = True
    )

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "embeddings" : embeddings,
                "paths"      : image_paths
            },
            f
        )

    print(f"\n  Saved  → {output_path}")
    print(f"  Images : {len(image_paths)}")
    print(f"  Shape  : {embeddings.shape}")


# ==================================================
# LOAD EMBEDDINGS
# ==================================================

def load_embeddings(
    pkl_path : str = OUTPUT_PKL
) -> tuple:

    if not os.path.exists(pkl_path):
        print(f"  No embeddings at {pkl_path}")
        return np.array([]), []

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    embeddings  = data["embeddings"]
    image_paths = data["paths"]

    print(f"  Loaded {len(image_paths)} embeddings")
    print(f"  Shape  : {embeddings.shape}")

    return embeddings, image_paths


# ==================================================
# SEARCH IMAGES BY TEXT — FIXED DIM MATCH
# ==================================================

def search_images_by_text(
    query        : str,
    embeddings   : np.ndarray,
    image_paths  : list,
    model,
    processor,
    top_k        : int = 5
) -> list:

    # Get text embedding — same model path as images
    text_emb = get_text_embedding(
        text      = query,
        model     = model,
        processor = processor
    )

    img_dim  = embeddings.shape[1]
    txt_dim  = text_emb.shape[0]

    print(f"  Image dim : {img_dim}")
    print(f"  Text dim  : {txt_dim}")

    # If dims don't match — project text to image dim
    if txt_dim != img_dim:
        print(f"  Dim mismatch! Projecting {txt_dim} → {img_dim}")

        # Pad or truncate to match
        if txt_dim < img_dim:
            text_emb = np.pad(
                text_emb,
                (0, img_dim - txt_dim),
                mode = "constant"
            )
        else:
            text_emb = text_emb[:img_dim]

    # Normalize
    text_emb = text_emb / (
        np.linalg.norm(text_emb) + 1e-8
    )

    norms           = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    norm_embeddings = embeddings / (norms + 1e-8)

    # Cosine similarity
    scores  = np.dot(norm_embeddings, text_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "image_path" : image_paths[idx],
            "image_name" : os.path.basename(image_paths[idx]),
            "score"      : round(float(scores[idx]), 4)
        }
        for idx in top_idx
    ]


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  IMAGE EMBEDDING PIPELINE")
    print("="*60)

    # Step 1
    print("\n[Step 1] Creating folders...")
    create_folders()

    # Step 2 — Check existing
    existing = 0
    if os.path.exists(IMAGE_FOLDER):
        existing = len([
            f for f in os.listdir(IMAGE_FOLDER)
            if os.path.splitext(f)[1].lower()
            in SUPPORTED_EXT
        ])

    if existing > 0:
        print(f"\n  Found {existing} existing images ✅")
    else:
        print("\n[Step 2] Extracting from PDFs...")
        extract_images_from_pdfs(
            data_folder = "data",
            output_dir  = IMAGE_FOLDER
        )

    # Step 3 — Count
    img_count = len([
        f for f in os.listdir(IMAGE_FOLDER)
        if os.path.splitext(f)[1].lower()
        in SUPPORTED_EXT
    ]) if os.path.exists(IMAGE_FOLDER) else 0

    print(f"\n  Images: {img_count}")

    if img_count == 0:
        print("  No images found!")
        sys.exit(1)

    # Step 4 — Generate
    print("\n[Step 3] Generating embeddings...")
    embeddings, image_paths = generate_image_embeddings(
        image_folder = IMAGE_FOLDER
    )

    if len(embeddings) == 0:
        print("  No embeddings generated!")
        sys.exit(1)

    # Step 5 — Save
    print("\n[Step 4] Saving...")
    save_embeddings(
        embeddings  = embeddings,
        image_paths = image_paths,
        output_path = OUTPUT_PKL
    )

    # Step 6 — Test search
    print("\n[Step 5] Testing search...")
    print("="*60)

    model, processor, _, _ = load_clip_model()

    for query in [
        "lung cancer xray",
        "breast cancer scan",
        "tumor pathology",
        "cancer cell image",
    ]:
        print(f"\nQuery: {query}")
        results = search_images_by_text(
            query       = query,
            embeddings  = embeddings,
            image_paths = image_paths,
            model       = model,
            processor   = processor,
            top_k       = 3
        )
        for i, res in enumerate(results):
            print(
                f"  [{i+1}] {res['image_name']}"
                f" (score: {res['score']})"
            )

    print("\n" + "="*60)
    print("  IMAGE EMBEDDINGS SAVED ✅")
    print("="*60)