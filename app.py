import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini

# HYBRID SEARCH IMPORTS
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 1. Load Environment Variables
load_dotenv()

# Safety Check
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# 2. Page Configuration
st.set_page_config(page_title="Benefits Navigator", layout="centered")

# 3. Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; font-family: 'Arial', sans-serif; }
    p, div, h1, h2, h3, h4, h5, h6, span, label, li { color: #333333 !important; }
    section[data-testid="stSidebar"] { background-color: #f4f4f4 !important; border-right: 2px solid #E60000; }
    section[data-testid="stSidebar"] * { color: #000000 !important; }
    h1, h2, h3 { color: #E60000 !important; }
    div[data-testid="stChatMessageContent"] { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; }
    div.stButton > button { background-color: #E60000; color: white !important; border: none; font-weight: bold; border-radius: 4px; }
    
    /* Specific styling for citations to ensure white text */
    div[data-testid="stExpanderDetails"] {
        background-color: #333333;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="stExpanderDetails"] * {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown("**AARP Foundation | Pennsylvania Pilot**")

st.divider()

# 5. Load the Hybrid Search Engine
@st.cache_resource
def load_hybrid_query_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model="models/gemini-3-flash-preview")
    
    vector_store = FaissVectorStore.from_persist_dir("storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir="storage"
    )
    policy_index = load_index_from_storage(storage_context=storage_context)

    vector_retriever = policy_index.as_retriever(similarity_top_k=5)
    
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=policy_index.docstore, 
        similarity_top_k=5
    )
    
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,  
        mode="reciprocal_rerank"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        system_prompt=(
            "You are a strict Medicaid Eligibility Assistant. "
            "You must use the provided documents only. "
            "If a checkbox is marked, state 'The box for [Title] is visually checked'. "
            "Always cite the source document and section header. "
            "CRITICAL: If the exact answer is not explicitly written in the provided context, "
            "you must reply ONLY with: 'I am unsure based on the provided documents.'"
        )
    )

try:
    query_engine = load_hybrid_query_engine()
except Exception as e:
    st.error("Knowledge base not found. Please run pipeline.py first.")
    st.stop()

# 6. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. The Chat Loop
if prompt := st.chat_input("Enter eligibility question here..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing structural policy data..."):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            
            with st.expander("View Source Citations"):
                for i, node in enumerate(response.source_nodes):
                    st.write(f"**Source {i+1}** (Fusion Score: {node.score:.4f})")
                    # Display the full text of the chunk
                    st.write(node.text)
                    st.divider()
                    
    st.session_state.messages.append({"role": "assistant", "content": response.response})

# Sidebar Info
with st.sidebar:
    st.markdown("### System Status")
    st.success("Hybrid Search Active")
    st.markdown("---")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()