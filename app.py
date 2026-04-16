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
    ::selection { background: #E60000; color: #FFFFFF !important; }
    div[data-testid="stChatMessageContent"] { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem; }
    div.stButton > button { background-color: #E60000; color: white !important; border: none; font-weight: bold; border-radius: 4px; }
    div.stButton > button:hover { background-color: #cc0000; color: white !important; }
    div[data-testid="stExpanderDetails"] { background-color: #333333; border-radius: 8px; padding: 15px; }
    div[data-testid="stExpanderDetails"] * { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# STATE CONFIG
# -----------------------------
STATE_MAP = {
    "Alabama": "AL",
    "Iowa": "IA",
    "Ohio": "OH",
    "Pennsylvania": "PA",
    "South Dakota": "SD",
}

CONDITION_OPTIONS = [
    "Intellectual/Developmental Disabilities",
    "Adults who are Ages 65+ or With Physical Disabilities",
    "Traumatic Brain and/or Spinal Cord Injuries",
    "Medically Fragile/Technology Dependent Children",
    "Mental Health",
    "Autism",
    "HIV/AIDS",
]

RELATIONSHIP_OPTIONS = [
    "Spouse",
    "Legal guardian",
    "Adult child",
    "Other relative",
    "Friend",
    "Other",
]

HOUSING_OPTIONS = [
    "I live with the person",
    "I do not live with the person but I live in the same state",
    "I do not live with the person and I do not live in the same state"
]

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown("**AARP Foundation | Multi-State Pilot**")

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
            "Always cite the source document, section header, and absolute page number as it appears in the full document (not the page number within the section). "
            "CRITICAL: If the exact answer is not explicitly written in the provided context, "
            "you must reply ONLY with: 'I am unsure based on the provided documents.'"
        )
    )

try:
    query_engine = load_hybrid_query_engine()
except Exception as e:
    st.error("Knowledge base not found. Please run pipeline.py first.")
    st.stop()

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "intake_submitted" not in st.session_state:
    st.session_state.intake_submitted = False

if "intake_data" not in st.session_state:
    st.session_state.intake_data = {}

if "selected_state_abbr" not in st.session_state:
    st.session_state.selected_state_abbr = None

# -----------------------------
# INTAKE FORM
# -----------------------------
st.subheader("Client Intake")

with st.form("client_intake_form"):
    st.markdown("### Current healthcare coverage")
    medicaid_status = st.radio("Is the person on or eligible for Medicaid?", ["Yes", "No"])
    
    st.markdown("### Location")
    selected_state_name = st.selectbox("State", options=list(STATE_MAP.keys()), index=0)
    congregate_setting = st.radio(
        "Does the person live in an assisted living facility or a group home?",
        ["Yes", "No", "Unknown"]
    )

    st.markdown("### Patient characteristics")
    age = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
    selected_conditions = st.multiselect(
        "Diseases and conditions: Check all that apply",
        CONDITION_OPTIONS
    )
    other_condition_checked = st.checkbox("Other condition")
    other_condition_text = ""
    if other_condition_checked:
        other_condition_text = st.text_input("Specify other condition")

    st.markdown("### Caregiver information")
    relationship = st.selectbox("What is your relationship to the person?", RELATIONSHIP_OPTIONS)
    lives_with_person = st.radio("Do you live with the person?", HOUSING_OPTIONS)
    submitted = st.form_submit_button("Save Intake Information")

if submitted:
    selected_state_abbr = STATE_MAP[selected_state_name]

    all_conditions = selected_conditions.copy()
    if other_condition_checked and other_condition_text.strip():
        all_conditions.append(f"Other: {other_condition_text.strip()}")

    st.session_state.intake_data = {
        "medicaid_status": medicaid_status,
        "state_name": selected_state_name,
        "state_abbr": selected_state_abbr,
        "congregate_setting": congregate_setting,
        "age": age,
        "conditions": all_conditions,
        "relationship": relationship,
        "lives_with_person": lives_with_person
    }
    st.session_state.intake_submitted = True

    if st.session_state.selected_state_abbr != selected_state_abbr:
        st.session_state.messages = []
        st.session_state.selected_state_abbr = selected_state_abbr

    st.success("Intake information saved.")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def build_case_summary(data):
    conditions_text = ", ".join(data["conditions"]) if data["conditions"] else "None reported"

    return f"""Case Summary:
- Is the person on or eligible for Medicaid? {data['medicaid_status']}.
- Location: {data['state_name']}
- Lives in assisted living facility or group home: {data['congregate_setting']}
- Age: {data['age']}
- Conditions: {conditions_text}
- Caregiver relationship: {data['relationship']}
- Caregiver lives with the person: {data['lives_with_person']}""".strip()


def build_eligibility_query(data):
    case_summary = build_case_summary(data)
    return f"""You are a Medicaid waiver navigation assistant.

Review the attached waiver and policy documents for {data['state_name']} only.

Goal:
Determine whether this person may be eligible for programs, waivers, or pathways that could allow a family caregiver to be paid, or that are relevant to paid family caregiving support.

Instructions:
1. Use only the attached documents.
2. Focus especially on waiver eligibility, caregiver eligibility, participant-directed options, consumer direction, self-direction, attendant care, personal assistance services, and any rules about legally responsible relatives, spouses, guardians, or live-in caregivers being paid.
3. If the person is not clearly eligible, explain what additional facts would be needed.
4. If Medicaid eligibility is a prerequisite, say that clearly.
5. Note any exclusions related to assisted living, group home residence, age, diagnosis category, or caregiver relationship.
6. Cite the specific document, section, and absolute page number (as it appears in the full document, not the page number within the section) as a footnote whenever it is referenced in the answer.
7. Be careful and precise: do not assume eligibility unless the documents support it.

{case_summary}

Please answer in this format:
1. Initial assessment of possible eligibility for paid family caregiver arrangements
2. Likely relevant waiver(s) or program(s)
3. Key reasons
4. Important restrictions or disqualifiers
5. Additional information needed
""".strip()


# -----------------------------
# ELIGIBILITY REVIEW SECTION
# -----------------------------
if st.session_state.intake_submitted:
    st.divider()
    st.subheader("Eligibility Review")

    case_summary = build_case_summary(st.session_state.intake_data)
    st.markdown("### Case Summary")
    st.markdown(case_summary.replace("\n", "  \n"))

    if st.button("Run Eligibility Review"):
        with st.chat_message("assistant"):
            with st.spinner("Reviewing waiver documents..."):
                eligibility_prompt = build_eligibility_query(st.session_state.intake_data)
                response = query_engine.query(eligibility_prompt)
                st.markdown(response.response)

# -----------------------------
# CHAT HISTORY + FOLLOW-UP
# -----------------------------
st.divider()
st.subheader("Ask a Follow-up Question")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter eligibility question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.intake_submitted:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing structural policy data..."):
                case_summary = build_case_summary(st.session_state.intake_data)
                full_prompt = (
                    f"State context: {st.session_state.intake_data['state_name']} "
                    f"({st.session_state.intake_data['state_abbr']}). "
                    f"{case_summary}\n\nQuestion: {prompt}"
                )
                response = query_engine.query(full_prompt)
                st.markdown(response.response)

        st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.error("Please complete the intake form first.")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### System Status")
    st.success("Hybrid Search Active")

    st.markdown("### Selected State")
    if st.session_state.intake_submitted:
        st.markdown(
            f"**{st.session_state.intake_data['state_name']} "
            f"({st.session_state.intake_data['state_abbr']})**"
        )
    else:
        st.markdown("Not selected yet")

    st.markdown("---")

    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Intake + Conversation"):
        st.session_state.messages = []
        st.session_state.intake_data = {}
        st.session_state.intake_submitted = False
        st.session_state.selected_state_abbr = None
        st.rerun()
