# src/chunking/chunk_pipeline.py

import os
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# FAST PDF LOADER
# ==================================================

def load_pdf(pdf_path: str) -> str:

    reader = PdfReader(pdf_path)

    def extract_page(page):
        try:
            return page.extract_text() or ""
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        pages = list(executor.map(extract_page, reader.pages))

    return "\n".join(pages)


# ==================================================
# LIGHTWEIGHT AGENTIC TITLE GENERATION
# ==================================================

def generate_chunk_title(chunk_text: str) -> str:

    lines = chunk_text.split("\n")

    for line in lines:

        line = line.strip()

        if 20 < len(line) < 120:
            return line[:100]

    return chunk_text[:80].replace("\n", " ")


# ==================================================
# CHUNKING
# ==================================================

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " "
        ]
    )

    chunks = splitter.split_text(text)

    results = []

    for idx, chunk in enumerate(chunks):

        results.append(
            {
                "chunk_id": idx,
                "title": generate_chunk_title(chunk),
                "text": chunk,
                "length": len(chunk)
            }
        )

    return results


# ==================================================
# PROCESS SINGLE PDF
# ==================================================

def process_pdf(
    pdf_path,
    chunk_size=1200,
    overlap=200
):

    print(f"\nProcessing: {os.path.basename(pdf_path)}")

    text = load_pdf(pdf_path)

    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(f"Chunks Generated: {len(chunks)}")

    return {
        "pdf_name": os.path.basename(pdf_path),
        "num_chunks": len(chunks),
        "chunks": chunks
    }


# ==================================================
# PROCESS ALL PDFS
# ==================================================

def process_pdf_folder(
    folder_path,
    chunk_size=1200,
    overlap=200
):

    pdf_files = sorted([
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(".pdf")
    ])

    all_results = []

    for pdf_file in pdf_files:

        result = process_pdf(
            pdf_file,
            chunk_size,
            overlap
        )

        all_results.append(result)

    return all_results


# ==================================================
# SAVE CHUNKS
# ==================================================

def save_chunks(results, output_folder="outputs/chunks"):

    os.makedirs(output_folder, exist_ok=True)

    for pdf in results:

        filename = (
            pdf["pdf_name"]
            .replace(".pdf", "_chunks.txt")
        )

        save_path = os.path.join(
            output_folder,
            filename
        )

        with open(
            save_path,
            "w",
            encoding="utf-8"
        ) as f:

            for chunk in pdf["chunks"]:

                f.write(
                    f"CHUNK ID : {chunk['chunk_id']}\n"
                )

                f.write(
                    f"TITLE : {chunk['title']}\n\n"
                )

                f.write(chunk["text"])

                f.write(
                    "\n\n"
                    + "=" * 100
                    + "\n\n"
                )

        print(f"Saved -> {save_path}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    DATA_FOLDER = "data"

    results = process_pdf_folder(
        folder_path=DATA_FOLDER,
        chunk_size=1200,
        overlap=200
    )

    save_chunks(results)

    total_chunks = sum(
        pdf["num_chunks"]
        for pdf in results
    )

    print("\n===================================")
    print("CHUNKING COMPLETED")
    print("===================================")
    print(f"PDFs Processed : {len(results)}")
    print(f"Total Chunks   : {total_chunks}")