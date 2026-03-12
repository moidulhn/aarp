import os
import re
import json
import time
import glob
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from google import genai

# =========================
# CONFIG
# =========================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

client = genai.Client(api_key=API_KEY)

INPUT_DIR = "docs"
OUTPUT_DIR = "waiver_outputs"
MODEL_NAME = "gemini-2.5-pro"

# Tune these as needed
MAX_CHARS_PER_CHUNK = 3500
CHUNK_OVERLAP = 300
MIN_KEYWORD_HITS = 2
TOP_N_CHUNKS = 15
SLEEP_BETWEEN_CALLS = 1.5

STATE_ABBREV_TO_NAME = {
    "AL": "Alabama",
    "PA": "Pennsylvania",
}

KEYWORDS = [
    "legally responsible",
    "legally responsible person",
    "legal liable relative",
    "relative",
    "relatives",
    "family caregiver",
    "guardian",
    "legal guardian",
    "spouse",
    "parent",
    "legally responsible individual",
    "personal care",
    "adult companion",
    "self-directed",
    "self directed",
    "consumer directed",
    "participant directed",
    "employer of record",
    "EOR",
    "extraordinary care",
    "payment may be made",
    "furnishing waiver services",
    "legally liable",
    "representative",
    "support broker",
    "provider qualifications",
]

TARGET_KEYS = [
    "state",
    "waiver_title",
    "source_file",
    "allows_payment_to_relatives",
    "allows_payment_to_legally_responsible_individuals",
    "eligible_relative_types",
    "ineligible_relative_types",
    "services_family_can_be_paid_for",
    "must_be_self_directed",
    "requires_extraordinary_care",
    "hour_limits",
    "other_conditions",
    "restrictions",
    "oversight_and_safeguards",
    "eor_or_representative_restrictions",
    "plain_language_summary",
    "navigator_answer",
    "citations",
    "confidence",
]

# =========================
# PDF / TEXT PROCESSING
# =========================

def infer_state_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    prefix = base[:2].upper()
    return STATE_ABBREV_TO_NAME.get(prefix, prefix)

def extract_pdf_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({
            "page_num": i + 1,
            "text": text.strip()
        })
    return pages

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        # Try not to cut mid-paragraph
        if end < n:
            breakpoint = text.rfind("\n\n", start, end)
            if breakpoint > start + max_chars // 2:
                end = breakpoint

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(end - overlap, start + 1)

    return chunks

def score_chunk(text: str, keywords: List[str]) -> Dict[str, Any]:
    text_lower = text.lower()
    hits = [kw for kw in keywords if kw.lower() in text_lower]
    return {
        "score": len(hits),
        "hits": hits
    }

def retrieve_relevant_chunks(
    pages: List[Dict[str, Any]],
    keywords: List[str],
    min_hits: int = MIN_KEYWORD_HITS
) -> List[Dict[str, Any]]:
    results = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(chunks, start=1):
            scored = score_chunk(chunk, keywords)
            if scored["score"] >= min_hits:
                results.append({
                    "page_num": page["page_num"],
                    "chunk_id": f"p{page['page_num']}_c{idx}",
                    "score": scored["score"],
                    "hits": scored["hits"],
                    "text": chunk,
                })

    results.sort(key=lambda x: (x["score"], -x["page_num"]), reverse=True)
    return results

# =========================
# GEMINI PROMPTS
# =========================

def build_extraction_prompt(
    state: str,
    waiver_title: str,
    source_file: str,
    chunks: List[Dict[str, Any]]
) -> str:
    joined_chunks = "\n\n".join(
        f"[PAGE {c['page_num']} | {c['chunk_id']} | keyword_hits={', '.join(c['hits'])}]\n{c['text']}"
        for c in chunks
    )

    return f"""
You are extracting Medicaid waiver policy information related ONLY to whether family caregivers,
relatives, legal guardians, spouses, parents, or other legally responsible individuals can be paid.

State: {state}
Waiver title: {waiver_title}
Source file: {source_file}

Return VALID JSON ONLY. No markdown fences. No explanatory text.

Use exactly these keys:
{{
  "state": "",
  "waiver_title": "",
  "source_file": "",
  "allows_payment_to_relatives": true,
  "allows_payment_to_legally_responsible_individuals": true,
  "eligible_relative_types": [],
  "ineligible_relative_types": [],
  "services_family_can_be_paid_for": [],
  "must_be_self_directed": true,
  "requires_extraordinary_care": true,
  "hour_limits": "",
  "other_conditions": [],
  "restrictions": [],
  "oversight_and_safeguards": [],
  "eor_or_representative_restrictions": [],
  "plain_language_summary": "",
  "navigator_answer": "",
  "citations": [],
  "confidence": ""
}}

Rules:
- Use ONLY the supplied text.
- Do NOT guess.
- If unclear, use null, "" or [] as appropriate.
- Distinguish carefully between:
  - relative
  - legal guardian
  - spouse
  - parent
  - legally responsible individual
  - legally liable relative
- Preserve service names as written in the source when possible.
- If the source text contains conditions or exceptions, include them.
- If the text appears inconsistent across sections, note that in restrictions or other_conditions.
- citations must be page/chunk references like "PAGE 144 p144_c1".
- navigator_answer should be 3 to 6 plain-English sentences written for a benefits navigator.

Relevant text:
{joined_chunks}
"""

def build_fallback_prompt(
    state: str,
    waiver_title: str,
    source_file: str,
    full_text_excerpt: str
) -> str:
    return f"""
You are extracting Medicaid waiver policy information related ONLY to whether family caregivers,
relatives, legal guardians, spouses, parents, or other legally responsible individuals can be paid.

State: {state}
Waiver title: {waiver_title}
Source file: {source_file}

The retrieved chunks may have missed some relevant language, so this is a fallback using a broader excerpt.

Return VALID JSON ONLY. No markdown fences. No explanatory text.

Use exactly these keys:
{{
  "state": "",
  "waiver_title": "",
  "source_file": "",
  "allows_payment_to_relatives": true,
  "allows_payment_to_legally_responsible_individuals": true,
  "eligible_relative_types": [],
  "ineligible_relative_types": [],
  "services_family_can_be_paid_for": [],
  "must_be_self_directed": true,
  "requires_extraordinary_care": true,
  "hour_limits": "",
  "other_conditions": [],
  "restrictions": [],
  "oversight_and_safeguards": [],
  "eor_or_representative_restrictions": [],
  "plain_language_summary": "",
  "navigator_answer": "",
  "citations": [],
  "confidence": ""
}}

Rules:
- Use ONLY the supplied text.
- Do NOT guess.
- If unclear, use null, "" or [] as appropriate.
- citations must be page references when possible.

Relevant text:
{full_text_excerpt}
"""

# =========================
# GEMINI CALLS
# =========================

def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def safe_parse_json(text: str) -> Dict[str, Any]:
    text = clean_json_text(text)
    return json.loads(text)

def normalize_output(obj: Dict[str, Any], state: str, waiver_title: str, source_file: str) -> Dict[str, Any]:
    normalized = {k: obj.get(k) for k in TARGET_KEYS}
    normalized["state"] = normalized.get("state") or state
    normalized["waiver_title"] = normalized.get("waiver_title") or waiver_title
    normalized["source_file"] = normalized.get("source_file") or source_file

    for list_key in [
        "eligible_relative_types",
        "ineligible_relative_types",
        "services_family_can_be_paid_for",
        "other_conditions",
        "restrictions",
        "oversight_and_safeguards",
        "eor_or_representative_restrictions",
        "citations",
    ]:
        if normalized[list_key] is None:
            normalized[list_key] = []
        elif not isinstance(normalized[list_key], list):
            normalized[list_key] = [str(normalized[list_key])]

    for str_key in [
        "hour_limits",
        "plain_language_summary",
        "navigator_answer",
        "confidence",
    ]:
        if normalized[str_key] is None:
            normalized[str_key] = ""

    return normalized

def call_gemini_json(prompt: str) -> Dict[str, Any]:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return safe_parse_json(response.text)

# =========================
# WAIVER PROCESSING
# =========================

def get_waiver_title_from_filename(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]

def build_broad_excerpt(pages: List[Dict[str, Any]], max_pages: int = 40, max_chars: int = 50000) -> str:
    combined = []
    for page in pages[:max_pages]:
        combined.append(f"[PAGE {page['page_num']}]\n{page['text']}")
    text = "\n\n".join(combined)
    return text[:max_chars]

def process_single_pdf(pdf_path: str) -> Dict[str, Any]:
    state = infer_state_from_filename(pdf_path)
    waiver_title = get_waiver_title_from_filename(pdf_path)
    source_file = os.path.basename(pdf_path)

    pages = extract_pdf_text_by_page(pdf_path)
    relevant_chunks = retrieve_relevant_chunks(pages, KEYWORDS, min_hits=MIN_KEYWORD_HITS)

    if relevant_chunks:
        selected_chunks = relevant_chunks[:TOP_N_CHUNKS]
        prompt = build_extraction_prompt(state, waiver_title, source_file, selected_chunks)
    else:
        broad_excerpt = build_broad_excerpt(pages)
        prompt = build_fallback_prompt(state, waiver_title, source_file, broad_excerpt)

    try:
        raw = call_gemini_json(prompt)
        result = normalize_output(raw, state, waiver_title, source_file)
    except Exception as e:
        result = {
            "state": state,
            "waiver_title": waiver_title,
            "source_file": source_file,
            "allows_payment_to_relatives": None,
            "allows_payment_to_legally_responsible_individuals": None,
            "eligible_relative_types": [],
            "ineligible_relative_types": [],
            "services_family_can_be_paid_for": [],
            "must_be_self_directed": None,
            "requires_extraordinary_care": None,
            "hour_limits": "",
            "other_conditions": [f"LLM extraction failed: {str(e)}"],
            "restrictions": [],
            "oversight_and_safeguards": [],
            "eor_or_representative_restrictions": [],
            "plain_language_summary": "",
            "navigator_answer": "",
            "citations": [],
            "confidence": "low",
        }

    result["_retrieved_chunk_count"] = len(relevant_chunks)
    result["_top_chunk_scores"] = [c["score"] for c in relevant_chunks[:TOP_N_CHUNKS]]
    return result

# =========================
# OUTPUT HELPERS
# =========================

def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = record.copy()
    for key, value in flat.items():
        if isinstance(value, list):
            flat[key] = " | ".join(map(str, value))
    return flat

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_results(records: List[Dict[str, Any]], output_dir: str) -> None:
    ensure_output_dir(output_dir)

    json_path = os.path.join(output_dir, "waiver_relative_payment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    flat_records = [flatten_for_csv(r) for r in records]
    df = pd.DataFrame(flat_records)

    preferred_cols = [
        "state",
        "waiver_title",
        "source_file",
        "allows_payment_to_relatives",
        "allows_payment_to_legally_responsible_individuals",
        "eligible_relative_types",
        "ineligible_relative_types",
        "services_family_can_be_paid_for",
        "must_be_self_directed",
        "requires_extraordinary_care",
        "hour_limits",
        "other_conditions",
        "restrictions",
        "oversight_and_safeguards",
        "eor_or_representative_restrictions",
        "plain_language_summary",
        "navigator_answer",
        "citations",
        "confidence",
        "_retrieved_chunk_count",
        "_top_chunk_scores",
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    df = df[existing_cols]

    csv_path = os.path.join(output_dir, "waiver_relative_payment_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to:  {csv_path}")

# =========================
# MAIN
# =========================

def main():
    ensure_output_dir(OUTPUT_DIR)

    pdf_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {INPUT_DIR}")

    all_results = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        result = process_single_pdf(pdf_path)
        all_results.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    save_results(all_results, OUTPUT_DIR)

if __name__ == "__main__":
    main()