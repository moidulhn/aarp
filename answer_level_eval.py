import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "models/gemini-2.5-flash-lite"
DEFAULT_EVAL_DIR = Path("eval")


JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "accuracy": {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER"},
                "feedback": {"type": "STRING"},
            },
            "required": ["score", "feedback"],
        },
        "completeness": {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER"},
                "feedback": {"type": "STRING"},
            },
            "required": ["score", "feedback"],
        },
        "relevance": {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER"},
                "feedback": {"type": "STRING"},
            },
            "required": ["score", "feedback"],
        },
        "clarity_readability": {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER"},
                "feedback": {"type": "STRING"},
            },
            "required": ["score", "feedback"],
        },
        "overall": {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER"},
                "feedback": {"type": "STRING"},
            },
            "required": ["score", "feedback"],
        },
    },
    "required": [
        "accuracy",
        "completeness",
        "relevance",
        "clarity_readability",
        "overall",
    ],
}


def extract_question(prompt: str) -> str:
    """Extract only the user's question from the stored app prompt."""
    match = re.search(
        r"Question:\s*(.*?)(?:\n\s*\nPlease answer|\n\s*\nCite sources|\Z)",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return prompt.strip()
    return match.group(1).strip()


def build_prometheus_prompt(
    *,
    question: str,
    answer: str,
    reference_answer: str,
) -> str:
    return f"""
You are a fair and strict evaluation judge.
Use a Prometheus-style rubric: compare the model answer against the user question and the reference answer, then assign integer scores from 1 to 5.

Evaluate all dimensions in one pass. Be strict about factual correctness. Do not reward fluent wording when the answer names the wrong waiver, program, state, eligibility rule, caregiver payment rule, or restriction.

Question:
{question}

Reference Answer:
{reference_answer}

Model Answer:
{answer}

Rubrics:

Accuracy:
5 = All factual claims are correct and consistent with the reference answer.
4 = Mostly accurate, with only minor imprecision.
3 = Mix of correct and incorrect or unsupported claims.
2 = Several important inaccuracies or misleading claims.
1 = Largely incorrect, names the wrong program/waiver, or contradicts the reference answer.

Completeness:
5 = Covers all key points needed to answer the question, including important conditions and restrictions.
4 = Covers most key points, with only minor omissions.
3 = Covers some key points but misses important details.
2 = Omits major information needed for a useful answer.
1 = Minimal or no useful coverage of the reference answer.

Relevance:
5 = Directly answers the question with no distracting information.
4 = Mostly relevant, with small amounts of unnecessary detail.
3 = Partially relevant but noticeably drifts from the question.
2 = Mostly loosely related or confusing.
1 = Off-topic.

Clarity and Readability:
5 = Clear, well organized, concise, and user-friendly.
4 = Generally clear, with minor wording or structure issues.
3 = Understandable but verbose, awkward, or uneven.
2 = Hard to follow.
1 = Very unclear.

Overall:
5 = Excellent answer that is accurate, complete, relevant, and clear.
4 = Good answer with minor issues.
3 = Usable but has meaningful gaps or risks.
2 = Poor answer with major issues.
1 = Not reliable.

Return only valid JSON matching the requested schema. Each feedback field must be 1 to 2 concise sentences.
""".strip()


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env before running.")
    return api_key


def coerce_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return -1
    return score if 1 <= score <= 5 else -1


def normalize_judge_result(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for dimension in [
        "accuracy",
        "completeness",
        "relevance",
        "clarity_readability",
        "overall",
    ]:
        item = raw.get(dimension) or {}
        normalized[dimension] = {
            "score": coerce_score(item.get("score")),
            "feedback": str(item.get("feedback", "")).strip(),
        }
    return normalized


def call_judge(
    *,
    client: genai.Client,
    model: str,
    question: str,
    answer: str,
    reference_answer: str,
) -> dict[str, Any]:
    prompt = build_prometheus_prompt(
        question=question,
        answer=answer,
        reference_answer=reference_answer,
    )
    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
            temperature=0,
            max_output_tokens=900,
        ),
    )

    parsed = getattr(result, "parsed", None)
    if parsed:
        if not isinstance(parsed, dict):
            parsed = dict(parsed)
        return normalize_judge_result(parsed)

    text = (getattr(result, "text", None) or "").strip()
    return normalize_judge_result(json.loads(text))


def evaluate_file(path: Path, client: genai.Client, model: str, overwrite: bool) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))

    existing_eval = data.get("answer_level_evaluation") or {}
    if existing_eval.get("status") == "completed" and not overwrite:
        return "skipped"

    prompt = str(data.get("prompt", ""))
    answer = str(data.get("answer", ""))
    reference_answer = str(data.get("reference answer", ""))

    if not prompt or not answer or not reference_answer:
        data["answer_level_evaluation"] = {
            "status": "failed",
            "error": "Missing prompt, answer, or reference answer.",
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return "failed"

    question = extract_question(prompt)
    scores = call_judge(
        client=client,
        model=model,
        question=question,
        answer=answer,
        reference_answer=reference_answer,
    )

    data["answer_level_evaluation"] = {
        "status": "completed",
        "style": "prometheus",
        "judge_model": model,
        "question": question,
        "scores": scores,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run answer-level Prometheus-style evaluation over eval JSON files."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--file", type=Path, default=None, help="Evaluate one JSON file only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())

    files = [args.file] if args.file else sorted(args.eval_dir.glob("*.json"))
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    for path in files:
        try:
            status = evaluate_file(path, client, args.model, args.overwrite)
        except Exception as exc:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["answer_level_evaluation"] = {
                "status": "failed",
                "style": "prometheus",
                "judge_model": args.model,
                "error": str(exc),
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            status = "failed"
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {path}")

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
