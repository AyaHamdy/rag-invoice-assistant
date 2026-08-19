import os
import json
from pypdf import PdfReader
import ollama

import re

def extract_total_fallback(text):
    """Find 'Total' label and the nearest dollar amount as a deterministic backup."""
    match = re.search(r"Total\s*:?\s*\$?([\d,]+\.\d{2})", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # fallback: last dollar amount in the document (often the total)
    amounts = re.findall(r"\$([\d,]+\.\d{2})", text)
    return float(amounts[-1].replace(",", "")) if amounts else None

folder = "Sample-Pdf-invoices"
records = []

EXTRACTION_PROMPT = """Extract invoice header fields from this invoice text. The text may be in a jumbled reading order — labels and values are not always adjacent. Match them by position order, not proximity: e.g. if you see "Subtotal Discount Shipping Total" as a group of labels followed later by "25.25 5.05 1.97 22.17" as a group of numbers, they correspond in the same order (Subtotal=25.25, Discount=5.05, Shipping=1.97, Total=22.17).

"customer" is the person/company under "Bill To", NOT the seller name (e.g. not "SuperStore").

Respond ONLY with JSON in this schema:
{"invoice_number": string, "customer": string, "date": string, "ship_mode": string, "total": number}

If a field is truly missing, use null."""

# EXTRACTION_PROMPT = """Extract invoice header fields from this invoice text.
# Respond ONLY with JSON in this schema:
# {"invoice_number": string, "customer": string, "date": string, "ship_mode": string, "total": number}

# If a field is missing, use null."""

for filename in os.listdir(folder):
    if not filename.endswith(".pdf"):
        continue

    reader = PdfReader(os.path.join(folder, filename))
    text = "\n".join(page.extract_text() for page in reader.pages)

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0},
        format="json",
    )

    try:
        parsed = json.loads(response["message"]["content"])
        if parsed.get("total") is None:
            parsed["total"] = extract_total_fallback(text)
        parsed["filename"] = filename
        parsed["full_text"] = text  # keep raw text too, for semantic search later
        records.append(parsed)
        print(f"OK: {filename} -> total={parsed.get('total')}")
    except json.JSONDecodeError:
        print(f"FAILED to parse: {filename}")
        print(response["message"]["content"])

# Save structured data
with open("invoices_structured.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\nExtracted {len(records)} invoices")