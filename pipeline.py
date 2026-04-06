import os
import warnings
import shutil
from dotenv import load_dotenv

# Use the stable standalone import
from llama_parse import LlamaParse 
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini
import faiss

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

def build_knowledge_base():
    if os.path.exists("storage"):
        shutil.rmtree("storage")

    print("Initializing the LlamaParse Engine")
    
    system_prompt = """
    This is a Medicaid Waiver document. 
    1. Extract tables precisely in Markdown format. 
    2. For checkboxes: map '☑' to '[X]' and '☐' to '[ ]'.
    3. Ensure Appendix B_1 tables keep row and column alignment.
    """

    parser = LlamaParse(
        result_type="markdown",
        system_prompt=system_prompt,
        verbose=True,
        num_workers=1
    )

    print("Parsing document")
    data_path = "data/2026-01-28-chc-1915c-appendix-b.pdf"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    documents = parser.load_data(data_path)

    print("Configuring Models with Thinking Config")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Enabled thinking mode for better reasoning on tables
    Settings.llm = Gemini(
        model="models/gemini-3-flash-preview",
    )

    print("Processing nodes")
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents)

    print("Building FAISS Vector Database")
    d = 384  
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context
    )
    
    index.storage_context.persist(persist_dir="storage")
    print("Knowledge Base built successfully")

if __name__ == "__main__":
    build_knowledge_base()