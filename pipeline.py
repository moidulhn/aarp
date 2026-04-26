import json
import os
import re
import warnings
import shutil
from collections import defaultdict
from pathlib import Path

import faiss
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from pypdf import PdfReader

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

PARSED_DIR = Path("data/parsed")
MANIFEST_PATH = Path("data/parse_manifest.json")

# Primary split: one `## PDF page N` section per PDF page (1-based index).
PDF_PAGE_HEADER_RE = re.compile(r"^## PDF page (\d+)\s*$", re.MULTILINE)
# Backward compatibility for older parsed files.
LEGACY_PAGE_COMMENT_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")

# If a page (or markdown subsection) is still larger than this, apply SentenceSplitter.
MAX_NODE_CHARS = 8000
SENTENCE_CHUNK_SIZE = 1536
SENTENCE_CHUNK_OVERLAP = 150


def get_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _coerce_positive_int(val: object, cap: int) -> int | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return max(1, min(val, cap)) if val > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit():
            v = int(s)
            return max(1, min(v, cap)) if v > 0 else None
        m = re.match(r"^(\d+)", s)
        if m:
            v = int(m.group(1))
            return max(1, min(v, cap)) if v > 0 else None
    return None


def infer_pdf_page_number(metadata: dict, sequential_index: int, pdf_page_count: int) -> int:
    """
    Map a LlamaParse segment to a 1-based PDF page index using parser metadata when present.
    Falls back to sequential position only when metadata has no usable page field.
    """
    if not metadata:
        return min(sequential_index + 1, pdf_page_count)

    for key in (
        "page",
        "page_number",
        "page_label",
        "start_page",
        "StartPage",
        "page_idx",
        "page_index",
    ):
        if key in metadata:
            coerced = _coerce_positive_int(metadata[key], pdf_page_count)
            if coerced is not None:
                return coerced

    nested = metadata.get("page_numbers") or metadata.get("pages")
    if isinstance(nested, (list, tuple)) and nested:
        coerced = _coerce_positive_int(nested[0], pdf_page_count)
        if coerced is not None:
            return coerced

    return min(sequential_index + 1, pdf_page_count)


def merge_llama_parse_docs_by_page(
    docs: list, pdf_page_count: int
) -> list[tuple[int, str]]:
    """Combine all parser segments that belong to the same PDF page."""
    buckets: dict[int, list[str]] = defaultdict(list)
    for i, doc in enumerate(docs):
        meta = getattr(doc, "metadata", None) or {}
        page = infer_pdf_page_number(meta, i, pdf_page_count)
        text = (getattr(doc, "text", None) or "").strip()
        if text:
            buckets[page].append(text)

    ordered_pages = sorted(buckets.keys())
    return [(p, "\n\n".join(buckets[p])) for p in ordered_pages]


def build_combined_markdown(page_blocks: list[tuple[int, str]]) -> str:
    """Write markdown with explicit page headers (used for chunking and human review)."""
    parts: list[str] = []
    for page_num, body in page_blocks:
        parts.append(f"## PDF page {page_num}\n\n{body}")
    return "\n\n".join(parts)


def _split_by_regex_positions(
    text: str, pattern: re.Pattern
) -> list[tuple[int, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    preamble = text[: matches[0].start()].strip()

    pages: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if i == 0 and preamble:
            body = f"{preamble}\n\n{body}".strip()
        pages.append((page_num, body))
    return pages


def split_combined_markdown_into_pages(text: str) -> list[tuple[int, str]]:
    """
    Split stored markdown into (pdf_page_1based, body) segments.
    Supports new `## PDF page N` headers and legacy `<!-- Page N -->` markers.
    """
    text = text.strip()
    if not text:
        return []

    new_style = _split_by_regex_positions(text, PDF_PAGE_HEADER_RE)
    if new_style:
        return new_style

    legacy = _split_by_regex_positions(text, LEGACY_PAGE_COMMENT_RE)
    if legacy:
        return legacy

    return [(1, text)]


def page_documents_from_parsed_markdown(
    md_text: str, file_name: str, state: str
) -> list[Document]:
    """One Document per PDF page so all indexed nodes carry the same page_number metadata."""
    segments = split_combined_markdown_into_pages(md_text)
    docs: list[Document] = []
    for page_num, body in segments:
        if not body.strip():
            continue
        docs.append(
            Document(
                text=body,
                metadata={
                    "file_name": file_name,
                    "state": state,
                    "page_number": page_num,
                },
            )
        )
    return docs


def build_nodes_with_page_aware_chunking(documents: list[Document]) -> list:
    """
    For each PDF page: split on markdown headers, then split oversized sections by sentence
    so no single retrieved node spans hundreds of pages.
    """
    md_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(
        chunk_size=SENTENCE_CHUNK_SIZE,
        chunk_overlap=SENTENCE_CHUNK_OVERLAP,
    )
    all_nodes: list = []

    for doc in documents:
        md_nodes = md_parser.get_nodes_from_documents([doc])
        for node in md_nodes:
            content = node.get_content()
            if len(content) <= MAX_NODE_CHARS:
                all_nodes.append(node)
                continue
            try:
                all_nodes.extend(sentence_splitter.get_nodes_from_node(node))
            except AttributeError:
                # Older llama_index: split via a synthetic Document preserving metadata.
                sub_docs = [
                    Document(
                        text=content,
                        metadata=dict(node.metadata),
                    )
                ]
                sub_nodes = sentence_splitter.get_nodes_from_documents(sub_docs)
                all_nodes.extend(sub_nodes)

    return all_nodes


def build_knowledge_base() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(Path("data").glob("*.pdf"))
    if not pdf_paths:
        print("Error: no PDFs found in data/")
        return

    manifest = load_manifest()
    updated_manifest = manifest.copy()

    system_prompt = """
    This is a Medicaid Waiver document.
    1. Extract tables precisely in Markdown format.
    2. For checkboxes: map '☑' to '[X]' and '☐' to '[ ]'.
    3. Ensure Appendix B_1 tables keep row and column alignment.
    """

    parser: LlamaParse | None = None

    for pdf_path in pdf_paths:
        page_count = get_page_count(pdf_path)
        cached = manifest.get(pdf_path.name)
        parsed_file = PARSED_DIR / f"{pdf_path.stem}.md"

        if cached == page_count and parsed_file.exists():
            print(f"Skipping (unchanged): {pdf_path.name}")
            continue

        if parser is None:
            print("Initializing the LlamaParse Engine")
            parser = LlamaParse(
                result_type="markdown",
                system_prompt=system_prompt,
                verbose=True,
                num_workers=1,
            )

        print(f"Parsing: {pdf_path.name} ({page_count} pages)")
        docs = parser.load_data(str(pdf_path))
        page_blocks = merge_llama_parse_docs_by_page(docs, page_count)
        combined = build_combined_markdown(page_blocks)
        parsed_file.write_text(combined, encoding="utf-8")
        updated_manifest[pdf_path.name] = page_count

    save_manifest(updated_manifest)

    print("Configuring models")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model="models/gemini-3-flash-preview")

    print("Loading parsed documents (one logical page per Document)")
    page_documents: list[Document] = []
    for pdf_path in pdf_paths:
        parsed_file = PARSED_DIR / f"{pdf_path.stem}.md"
        if not parsed_file.exists():
            continue
        md_text = parsed_file.read_text(encoding="utf-8")
        page_documents.extend(
            page_documents_from_parsed_markdown(
                md_text,
                file_name=pdf_path.name,
                state=pdf_path.name[:2].upper(),
            )
        )

    if not page_documents:
        print("Error: no parsed markdown found under data/parsed/")
        return

    print("Processing nodes (markdown splits within each page, then size cap)")
    nodes = build_nodes_with_page_aware_chunking(page_documents)

    print("Building FAISS Vector Database")
    if os.path.exists("storage"):
        shutil.rmtree("storage")

    faiss_index = faiss.IndexFlatL2(384)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir="storage")
    print("Knowledge Base built successfully")


if __name__ == "__main__":
    build_knowledge_base()
