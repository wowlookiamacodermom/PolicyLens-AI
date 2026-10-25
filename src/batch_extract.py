
import json
from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR.resolve()}")
        return

    for pdf_path in pdf_files:
        pages = []

        try:
            with fitz.open(pdf_path) as doc:
                for page_number, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    pages.append({
                        "page": page_number,
                        "text": text
                    })

            result = {
                "source_file": pdf_path.name,
                "page_count": len(pages),
                "pages": pages
            }

            output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            empty_pages = sum(not p["text"] for p in pages)
            print(
                f"Processed: {pdf_path.name} | "
                f"{len(pages)} pages | "
                f"{empty_pages} pages without extractable text"
            )

        except Exception as exc:
            print(f"Could not process {pdf_path.name}: {exc}")

    print(f"\nFinished. Check {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
