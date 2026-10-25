
import csv
import json
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/annotations/clauses_to_label.csv")


def split_into_blocks(text):
    # Split extracted text into paragraph-like candidate clauses.
    paragraphs = re.split(r"\n\s*\n", text)

    blocks = []
    for paragraph in paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()

        # Ignore tiny fragments such as isolated page numbers.
        if len(cleaned) >= 40:
            blocks.append(cleaned)

    return blocks


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json_files = sorted(PROCESSED_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {PROCESSED_DIR.resolve()}")
        print("Run the PDF extraction script first.")
        return

    rows = []

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            for page_data in data.get("pages", []):
                page_number = page_data.get("page")
                page_text = page_data.get("text", "")

                for block in split_into_blocks(page_text):
                    rows.append({
                        "policy_id": json_file.stem,
                        "source_file": data.get(
                            "source_file", json_file.stem + ".pdf"
                        ),
                        "page": page_number,
                        "text": block,
                        "label": "",
                        "reviewed": "no"
                    })

        except Exception as exc:
            print(f"Could not read {json_file.name}: {exc}")

    columns = [
        "policy_id", "source_file", "page",
        "text", "label", "reviewed"
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidate clause blocks created: {len(rows)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print("Labels are blank intentionally. Review clauses before training.")


if __name__ == "__main__":
    main()
