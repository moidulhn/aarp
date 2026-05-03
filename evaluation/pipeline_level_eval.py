import argparse
import math
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
DEFAULT_EVAL_DIR = Path("eval")


def extract_question(prompt: str) -> str:
    """Extract only the user question from the stored app prompt."""
    match = re.search(
        r"Question:\s*(.*?)(?:\n\s*\nPlease answer|\n\s*\nCite sources|\Z)",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return prompt.strip()
    return match.group(1).strip()


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env before running.")
    return api_key


def load_ragas_components():
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS dependencies are not installed. Install them first, for example:\n"
            "  pip install ragas datasets langchain-google-genai\n"
            "Then rerun this script."
        ) from exc

    try:
        from ragas.metrics import context_precision, context_recall, faithfulness

        metric_factory = lambda: [
            context_precision,
            context_recall,
            faithfulness,
        ]
    except ImportError as exc:
        raise RuntimeError(
            "Installed RAGAS version does not expose the classic metric objects "
            "context_precision, context_recall, and faithfulness."
        ) from exc

    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai is not installed. Install it first:\n"
            "  pip install langchain-google-genai"
        ) from exc

    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

    return {
        "Dataset": Dataset,
        "evaluate": evaluate,
        "metric_factory": metric_factory,
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "GoogleGenerativeAIEmbeddings": GoogleGenerativeAIEmbeddings,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
    }


def build_langchain_ragas_models(runtime: dict[str, Any], model: str, embedding_model: str):
    api_key = get_api_key()
    # LangChain's Gemini wrapper expects names like "gemini-2.5-flash-lite",
    # while google.genai examples often use "models/gemini-...".
    model = model.removeprefix("models/")
    embedding_model = embedding_model.removeprefix("models/")
    judge_llm = runtime["LangchainLLMWrapper"](
        runtime["ChatGoogleGenerativeAI"](
            model=model,
            google_api_key=api_key,
            temperature=0,
        )
    )
    judge_embeddings = runtime["LangchainEmbeddingsWrapper"](
        runtime["GoogleGenerativeAIEmbeddings"](
            model=embedding_model,
            google_api_key=api_key,
        )
    )
    return judge_llm, judge_embeddings


def is_number(value: Any) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(value) or math.isinf(value))


def has_valid_pipeline_scores(existing_eval: dict[str, Any]) -> bool:
    if existing_eval.get("status") != "completed":
        return False
    metrics = existing_eval.get("metrics") or {}
    return all(
        is_number(metrics.get(name))
        for name in ["context_precision", "context_recall", "faithfulness"]
    )


def get_context_texts(data: dict[str, Any], max_contexts: int, max_context_chars: int) -> list[str]:
    contexts: list[str] = []
    for item in data.get("retrieved_chunks") or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        metadata = item.get("metadata") or {}
        prefix = (
            f"Source: {metadata.get('file_name')}, "
            f"page {metadata.get('page_number')}, "
            f"state {metadata.get('state')}\n"
        )
        contexts.append(prefix + text[:max_context_chars])
        if len(contexts) >= max_contexts:
            break
    return contexts


def get_score(result: Any, metric_names: list[str]) -> float | None:
    if hasattr(result, "to_pandas"):
        row = result.to_pandas().iloc[0].to_dict()
    elif isinstance(result, dict):
        row = result
    else:
        row = dict(result)

    normalized = {str(k).lower(): v for k, v in row.items()}
    for name in metric_names:
        key = name.lower()
        if key in normalized:
            try:
                value = float(normalized[key])
                if math.isnan(value) or math.isinf(value):
                    return None
                return round(value, 4)
            except (TypeError, ValueError):
                return None
    return None


def run_ragas_for_file(
    *,
    data: dict[str, Any],
    Dataset,
    evaluate,
    metrics,
    judge_llm,
    judge_embeddings,
    max_contexts: int,
    max_context_chars: int,
) -> dict[str, Any]:
    question = extract_question(str(data.get("prompt", "")))
    answer = str(data.get("answer", ""))
    reference = str(data.get("reference answer", ""))
    contexts = get_context_texts(data, max_contexts, max_context_chars)

    if not question or not answer or not reference or not contexts:
        raise RuntimeError("Missing question, answer, reference answer, or retrieved contexts.")

    modern_data = {
        "user_input": [question],
        "response": [answer],
        "reference": [reference],
        "retrieved_contexts": [contexts],
    }

    try:
        dataset = Dataset.from_dict(modern_data)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=True,
        )
    except TypeError:
        result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)
    except Exception:
        legacy_data = {
            "question": [question],
            "answer": [answer],
            "ground_truth": [reference],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(legacy_data)
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings,
                raise_exceptions=True,
            )
        except TypeError:
            result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

    scores = {
        "context_precision": get_score(
            result,
            [
                "context_precision",
                "llm_context_precision_with_reference",
                "ContextPrecision",
            ],
        ),
        "context_recall": get_score(
            result,
            [
                "context_recall",
                "ContextRecall",
                "llm_context_recall",
            ],
        ),
        "faithfulness": get_score(
            result,
            [
                "faithfulness",
                "Faithfulness",
            ],
        ),
        "question": question,
        "num_contexts": len(contexts),
    }
    missing = [
        name
        for name in ["context_precision", "context_recall", "faithfulness"]
        if scores[name] is None
    ]
    if missing:
        raise RuntimeError(f"RAGAS returned no numeric score for: {', '.join(missing)}")
    return scores


def insert_after_key(data: dict[str, Any], key: str, new_key: str, value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    inserted = False
    for current_key, current_value in data.items():
        if current_key == new_key:
            if not inserted:
                output[new_key] = value
                inserted = True
            continue
        output[current_key] = current_value
        if current_key == key:
            output[new_key] = value
            inserted = True
    if not inserted:
        output[new_key] = value
    return output


def evaluate_file(
    *,
    path: Path,
    Dataset,
    evaluate,
    metrics,
    judge_llm,
    judge_embeddings,
    model: str,
    max_contexts: int,
    max_context_chars: int,
    overwrite: bool,
) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    existing_eval = data.get("pipeline_level_evaluation") or {}
    if has_valid_pipeline_scores(existing_eval) and not overwrite:
        return "skipped"

    scores = run_ragas_for_file(
        data=data,
        Dataset=Dataset,
        evaluate=evaluate,
        metrics=metrics,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        max_contexts=max_contexts,
        max_context_chars=max_context_chars,
    )

    payload = {
        "status": "completed",
        "framework": "ragas",
        "judge_model": model,
        "metrics": {
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
            "faithfulness": scores["faithfulness"],
        },
        "question": scores["question"],
        "num_contexts": scores["num_contexts"],
        "max_context_chars": max_context_chars,
    }
    updated = insert_after_key(data, "answer_level_evaluation", "pipeline_level_evaluation", payload)
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return "completed"


def write_failure(path: Path, model: str, error: Exception) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "status": "failed",
        "framework": "ragas",
        "judge_model": model,
        "error": str(error),
    }
    updated = insert_after_key(data, "answer_level_evaluation", "pipeline_level_evaluation", payload)
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS pipeline-level evaluation over eval JSON files."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--model", default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument("--file", type=Path, default=None, help="Evaluate one JSON file only.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=2500)
    args = parser.parse_args()

    runtime = load_ragas_components()
    Dataset = runtime["Dataset"]
    evaluate = runtime["evaluate"]
    metrics = runtime["metric_factory"]()
    judge_llm, judge_embeddings = build_langchain_ragas_models(
        runtime, args.model, args.embedding_model
    )

    files = [args.file] if args.file else sorted(args.eval_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for path in files:
        try:
            status = evaluate_file(
                path=path,
                Dataset=Dataset,
                evaluate=evaluate,
                metrics=metrics,
                judge_llm=judge_llm,
                judge_embeddings=judge_embeddings,
                model=args.model,
                max_contexts=args.max_contexts,
                max_context_chars=args.max_context_chars,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            write_failure(path, args.model, exc)
            status = "failed"

        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {path}")

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
