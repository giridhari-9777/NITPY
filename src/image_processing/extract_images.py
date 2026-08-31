# src/image_processing/extract_images.py

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# CONFIG
# ==================================================

DATA_FOLDER       = "data"
OUTPUT_IMAGES     = "outputs/extracted_images"
OUTPUT_PROCESSED  = "outputs/processed_images"
OUTPUT_METADATA   = "outputs/image_metadata"
SUPPORTED_EXT     = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
MIN_IMAGE_SIZE    = 32     # pixels
MAX_IMAGE_SIZE    = 1024   # pixels
MIN_FILE_SIZE     = 5000   # bytes


# ==================================================
# CREATE FOLDERS
# ==================================================

def create_folders():

    folders = [
        "outputs",
        "outputs/extracted_images",
        "outputs/processed_images",
        "outputs/image_metadata",
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("  Folders ready ✅")


# ==================================================
# EXTRACT IMAGES FROM SINGLE PDF
# ==================================================

def extract_from_pdf(
    pdf_path   : str,
    output_dir : str = OUTPUT_IMAGES
) -> list:

    try:
        import fitz

        os.makedirs(output_dir, exist_ok=True)

        pdf_name  = Path(pdf_path).stem
        doc       = fitz.open(pdf_path)
        extracted = []

        for page_num in range(len(doc)):

            page = doc[page_num]

            try:
                # Render page as image
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)

                img_filename = (
                    f"{pdf_name}_page{page_num+1}.png"
                )
                img_path = os.path.join(
                    output_dir, img_filename
                )

                pix.save(img_path)

                # Validate
                if os.path.getsize(img_path) > MIN_FILE_SIZE:
                    extracted.append({
                        "image_path"  : img_path,
                        "image_name"  : img_filename,
                        "source_pdf"  : os.path.basename(pdf_path),
                        "page_number" : page_num + 1,
                        "extracted_at": datetime.now().isoformat()
                    })
                else:
                    os.remove(img_path)

            except Exception:
                continue

        doc.close()
        return extracted

    except ImportError:
        print("  Run: pip install PyMuPDF")
        return []

    except Exception as e:
        print(f"  PDF error: {e}")
        return []


# ==================================================
# EXTRACT FROM ALL PDFs
# ==================================================

def extract_all_pdfs(
    data_folder : str = DATA_FOLDER,
    output_dir  : str = OUTPUT_IMAGES
) -> list:

    pdf_files = sorted([
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"  No PDFs in {data_folder}")
        return []

    print(f"\nExtracting from {len(pdf_files)} PDFs...")
    print("="*60)

    all_images = []

    for i, pdf_path in enumerate(pdf_files):

        print(
            f"\n[{i+1}/{len(pdf_files)}] "
            f"{os.path.basename(pdf_path)}"
        )

        images = extract_from_pdf(
            pdf_path   = pdf_path,
            output_dir = output_dir
        )

        all_images.extend(images)
        print(f"  → {len(images)} images extracted")

    print(f"\n{'='*60}")
    print(f"Total images extracted: {len(all_images)} ✅")

    return all_images


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
            img_cv = cv2.cvtColor(
                img_cv, cv2.COLOR_BGR2RGB
            )
            from PIL import Image
            return Image.fromarray(img_cv)
    except Exception:
        pass

    return None


# ==================================================
# PROCESS SINGLE IMAGE
# ==================================================

def process_image(
    image_path  : str,
    output_dir  : str = OUTPUT_PROCESSED,
    target_size : int = 512
) -> dict:

    from PIL import Image

    try:
        # Load
        img = safe_load_image(image_path)

        if img is None:
            return {}

        # Validate size
        if (img.width  < MIN_IMAGE_SIZE or
            img.height < MIN_IMAGE_SIZE):
            return {}

        original_size = (img.width, img.height)

        # Resize — keep aspect ratio
        if img.width > MAX_IMAGE_SIZE or img.height > MAX_IMAGE_SIZE:
            img.thumbnail(
                (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
                Image.LANCZOS
            )

        # Get image stats
        img_array = np.array(img)

        mean_brightness = float(np.mean(img_array))
        std_contrast    = float(np.std(img_array))

        # Skip very dark or uniform images
        if mean_brightness < 10:
            return {}

        if std_contrast < 5:
            return {}

        # Save processed image
        os.makedirs(output_dir, exist_ok=True)

        filename  = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)

        img.save(save_path, "PNG", optimize=True)

        return {
            "original_path"   : image_path,
            "processed_path"  : save_path,
            "image_name"      : filename,
            "original_size"   : original_size,
            "processed_size"  : (img.width, img.height),
            "mean_brightness" : round(mean_brightness, 2),
            "std_contrast"    : round(std_contrast,    2),
            "processed_at"    : datetime.now().isoformat()
        }

    except Exception as e:
        print(f"  Process error: {e}")
        return {}


# ==================================================
# PROCESS ALL IMAGES
# ==================================================

def process_all_images(
    image_folder : str = OUTPUT_IMAGES,
    output_dir   : str = OUTPUT_PROCESSED
) -> list:

    if not os.path.exists(image_folder):
        print(f"  Folder not found: {image_folder}")
        return []

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not image_files:
        print(f"  No images in {image_folder}")
        return []

    print(f"\nProcessing {len(image_files)} images...")
    print("="*60)

    processed = []
    failed    = 0

    for i, image_path in enumerate(image_files):

        result = process_image(
            image_path = image_path,
            output_dir = output_dir
        )

        if result:
            processed.append(result)
        else:
            failed += 1

        if (i + 1) % 200 == 0:
            print(
                f"  Processed {i+1}/{len(image_files)}"
                f" (failed: {failed})"
            )

    print(f"\n  Success : {len(processed)}")
    print(f"  Failed  : {failed}")

    return processed


# ==================================================
# SAVE IMAGE METADATA
# ==================================================

def save_metadata(
    metadata    : list,
    output_path : str = "outputs/image_metadata/metadata.pkl"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n  Saved metadata → {output_path}")
    print(f"  Total images   : {len(metadata)}")


# ==================================================
# LOAD IMAGE METADATA
# ==================================================

def load_metadata(
    pkl_path : str = "outputs/image_metadata/metadata.pkl"
) -> list:

    if not os.path.exists(pkl_path):
        print(f"  No metadata at {pkl_path}")
        return []

    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"  Loaded {len(metadata)} image records")

    return metadata


# ==================================================
# GET IMAGE STATS
# ==================================================

def get_image_stats(metadata: list) -> dict:

    if not metadata:
        return {}

    sizes       = [m["processed_size"] for m in metadata]
    widths      = [s[0] for s in sizes]
    heights     = [s[1] for s in sizes]
    brightness  = [m["mean_brightness"] for m in metadata]
    contrast    = [m["std_contrast"]    for m in metadata]

    # Count by source PDF
    by_pdf = {}
    for m in metadata:
        src = m.get("image_name", "").split("_page")[0]
        if src not in by_pdf:
            by_pdf[src] = 0
        by_pdf[src] += 1

    return {
        "total_images"    : len(metadata),
        "avg_width"       : round(float(np.mean(widths)),     1),
        "avg_height"      : round(float(np.mean(heights)),    1),
        "avg_brightness"  : round(float(np.mean(brightness)), 2),
        "avg_contrast"    : round(float(np.mean(contrast)),   2),
        "by_pdf"          : by_pdf
    }


# ==================================================
# PRINT STATS
# ==================================================

def print_stats(stats: dict):

    print(f"\n{'='*60}")
    print(f"  IMAGE PROCESSING STATS")
    print(f"{'='*60}")
    print(f"  Total Images    : {stats['total_images']}")
    print(f"  Avg Width       : {stats['avg_width']} px")
    print(f"  Avg Height      : {stats['avg_height']} px")
    print(f"  Avg Brightness  : {stats['avg_brightness']}")
    print(f"  Avg Contrast    : {stats['avg_contrast']}")

    print(f"\n  Images by PDF:")
    for pdf, count in list(stats["by_pdf"].items())[:5]:
        print(f"     {pdf[:40]} : {count}")

    if len(stats["by_pdf"]) > 5:
        print(
            f"     ... and "
            f"{len(stats['by_pdf'])-5} more"
        )

    print(f"{'='*60}\n")


# ==================================================
# IMAGE PROCESSOR CLASS
# ==================================================

class ImageProcessor:

    def __init__(self):

        print("\nInitializing Image Processor...")
        create_folders()
        self.metadata = []
        print("Image Processor ready!\n")


    def extract(
        self,
        data_folder : str = DATA_FOLDER,
        output_dir  : str = OUTPUT_IMAGES
    ) -> list:

        return extract_all_pdfs(
            data_folder = data_folder,
            output_dir  = output_dir
        )


    def process(
        self,
        image_folder : str = OUTPUT_IMAGES,
        output_dir   : str = OUTPUT_PROCESSED
    ) -> list:

        return process_all_images(
            image_folder = image_folder,
            output_dir   = output_dir
        )


    def save(
        self,
        metadata    : list,
        output_path : str = "outputs/image_metadata/metadata.pkl"
    ):
        save_metadata(metadata, output_path)
        self.metadata = metadata


    def load(
        self,
        pkl_path : str = "outputs/image_metadata/metadata.pkl"
    ) -> list:
        self.metadata = load_metadata(pkl_path)
        return self.metadata


    def stats(self) -> dict:

        if not self.metadata:
            print("  No metadata loaded!")
            return {}

        s = get_image_stats(self.metadata)
        print_stats(s)
        return s


    def run_full_pipeline(self):

        print("\n" + "="*60)
        print("  FULL IMAGE PROCESSING PIPELINE")
        print("="*60)

        # Step 1 — Extract
        print("\n[Step 1] Extracting images from PDFs...")
        extracted = self.extract()

        if not extracted:
            print("  No images extracted!")
            return

        # Step 2 — Process
        print("\n[Step 2] Processing images...")
        processed = self.process()

        if not processed:
            print("  No images processed!")
            return

        # Step 3 — Save metadata
        print("\n[Step 3] Saving metadata...")
        self.save(processed)

        # Step 4 — Stats
        print("\n[Step 4] Stats...")
        self.stats()

        print("\n" + "="*60)
        print("  IMAGE PROCESSING COMPLETE ✅")
        print("="*60)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    processor = ImageProcessor()

    # Check if already extracted
    existing = 0
    if os.path.exists(OUTPUT_IMAGES):
        existing = len([
            f for f in os.listdir(OUTPUT_IMAGES)
            if os.path.splitext(f)[1].lower()
            in SUPPORTED_EXT
        ])

    if existing > 0:
        print(f"\nFound {existing} existing images")
        print("Skipping extraction, processing only...")

        # Just process existing images
        processed = process_all_images()

        if processed:
            save_metadata(processed)
            stats = get_image_stats(processed)
            print_stats(stats)

    else:
        # Run full pipeline
        processor.run_full_pipeline()