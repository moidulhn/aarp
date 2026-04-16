import os
import json
import warnings
import shutil
from pathlib import Path
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import StorageContext, Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini
from pypdf import PdfReader
import faiss

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

PARSED_DIR = Path("data/parsed")
MANIFEST_PATH = Path("data/parse_manifest.json")

def get_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def build_knowledge_base():
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = list(Path("data").glob("*.pdf"))
    if not pdf_paths:
        print("Error: no PDFs found in data/")
        return

    manifest = load_manifest()

    system_prompt = """
    This is a Medicaid Waiver document.
    1. Extract tables precisely in Markdown format.
    2. For checkboxes: map '☑' to '[X]' and '☐' to '[ ]'.
    3. Ensure Appendix B_1 tables keep row and column alignment.
    """

    parser = None
    updated_manifest = manifest.copy()

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
                num_workers=1
            )

        print(f"Parsing: {pdf_path.name} ({page_count} pages)")
        docs = parser.load_data(str(pdf_path))
        pages = [f"<!-- Page {i+1} -->\n{doc.text}" for i, doc in enumerate(docs)]
        combined = "\n\n".join(pages)
        parsed_file.write_text(combined, encoding="utf-8")
        updated_manifest[pdf_path.name] = page_count

    save_manifest(updated_manifest)

    print("Configuring models")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model="models/gemini-3-flash-preview")

    print("Loading parsed documents")
    documents = []
    for pdf_path in pdf_paths:
        parsed_file = PARSED_DIR / f"{pdf_path.stem}.md"
        if parsed_file.exists():
            documents.append(Document(
                text=parsed_file.read_text(encoding="utf-8"),
                metadata={"file_name": pdf_path.name}
            ))

    print("Processing nodes")
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)

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
