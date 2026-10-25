"""Extract text from a PDF while preserving 1-indexed page numbers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import fitz

def extract_pdf(path: str | Path) -> list[dict]:
    records = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            records.append({"page": idx + 1, "text": text})
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pages = extract_pdf(args.pdf)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source_file": Path(args.pdf).name, "pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")
    empty = sum(not p["text"] for p in pages)
    print(f"Extracted {len(pages)} pages; {empty} pages have no selectable text. OCR may be needed.")
    print(f"Saved: {out}")
if __name__ == "__main__":
    main()
