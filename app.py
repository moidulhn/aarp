import streamlit as st
import os
import glob
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load Environment Variables
load_dotenv()

# Safety Check
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize the Client
client = genai.Client(api_key=api_key)

# 2. Page Configuration (Clean, No Icons)
st.set_page_config(page_title="Medicaid Benefits Navigation", layout="centered")

# 3. Custom CSS for AARP Styling & Visibility Fixes
st.markdown("""
    <style>
    /* 1. Global Reset */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    
    /* 2. Text Colors - Force Dark Gray globally */
    p, div, h1, h2, h3, h4, h5, h6, span, label, li {
        color: #333333 !important;
    }

    /* 3. Sidebar Specifics */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4 !important;
        border-right: 2px solid #E60000;
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* 4. AARP Red Highlights */
    h1, h2, h3 {
        color: #E60000 !important;
    }
    ::selection {
        background: #E60000;
        color: #FFFFFF !important;
    }

    /* 5. Chat Bubbles */
    div[data-testid="stChatMessageContent"] {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #E60000;
        color: white !important;
        border: none;
        font-weight: bold;
        border-radius: 4px;
    }
    div.stButton > button:hover {
        background-color: #cc0000;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# STATE SELECTION CONFIG
# -----------------------------
STATE_MAP = {
    "Pennsylvania": "PA",
    "Alabama": "AL",
}

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown("**AARP Foundation | Multi-State Pilot**")

st.divider()

# -----------------------------
# STATE SELECTOR
# -----------------------------
selected_state_name = st.selectbox(
    "Select the state where the person receiving services lives:",
    options=list(STATE_MAP.keys()),
    index=0
)
selected_state_abbr = STATE_MAP[selected_state_name]

# 5. Multimodal File Handling
@st.cache_resource
def load_docs_to_gemini(state_abbr):
    """
    Scans 'docs/' folder, filters PDFs by state abbreviation using the first
    2 characters of the filename, then checks if files are already uploaded
    to Gemini to avoid re-uploading.
    
    Example filenames:
      PA_waiver1.pdf
      AL_familycaregiver.pdf
    """
    uploaded_files = []
    all_doc_paths = glob.glob("docs/*.pdf")

    # Filter to files whose first two filename chars match state abbreviation
    doc_paths = [
        path for path in all_doc_paths
        if os.path.basename(path)[:2].upper() == state_abbr.upper()
    ]

    if not doc_paths:
        st.sidebar.error(f"No PDF files found in 'docs/' folder for state: {state_abbr}")
        return []

    status_bar = st.sidebar.status(f"System Status: Connecting to Gemini for {state_abbr}...")

    # Step 1: List files already on Google's server
    existing_files_map = {}
    try:
        for f in client.files.list():
            existing_files_map[f.display_name] = f
    except Exception as e:
        st.sidebar.warning(f"Could not list existing files: {e}")

    # Step 2: Upload only if missing
    for path in doc_paths:
        file_name = os.path.basename(path)
        status_bar.write(f"Processing: {file_name}")
        
        try:
            if file_name in existing_files_map:
                status_bar.write(f"✅ Found cached: {file_name}")
                uploaded_files.append(existing_files_map[file_name])
            else:
                status_bar.write(f"⬆️ Uploading: {file_name}...")
                with open(path, "rb") as f:
                    sample_file = client.files.upload(
                        file=f,
                        config={
                            "display_name": file_name,
                            "mime_type": "application/pdf"
                        }
                    )
                uploaded_files.append(sample_file)
                time.sleep(1)
                
        except Exception as e:
            st.sidebar.error(f"Failed to load {file_name}: {e}")
            
    status_bar.update(label=f"System Ready ({state_abbr})", state="complete", expanded=False)
    return uploaded_files

# Load only the selected state's files
docs_context = load_docs_to_gemini(selected_state_abbr)

# Optional: clear conversation when state changes
if "selected_state_abbr" not in st.session_state:
    st.session_state.selected_state_abbr = selected_state_abbr
elif st.session_state.selected_state_abbr != selected_state_abbr:
    st.session_state.selected_state_abbr = selected_state_abbr
    st.session_state.messages = []
    st.rerun()

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

    if docs_context:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing policy documents...")
            
            try:
                content_payload = [
                    f"You are a strict Medicaid Eligibility Assistant for {selected_state_name} ({selected_state_abbr}).",
                    "Use only the attached policy documents to answer.",
                    "Cite the specific document name or section when possible.",
                    "If the answer depends on a checked box, explicitly state that.",
                    f"Only rely on documents for {selected_state_name}.",
                ]
                content_payload.extend(docs_context)
                content_payload.append(prompt)

                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=content_payload
                )
                
                message_placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                if "429" in str(e):
                    message_placeholder.warning("High traffic. Retrying in 5 seconds...")
                    time.sleep(5)
                    try:
                        response = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=content_payload
                        )
                        message_placeholder.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception:
                        message_placeholder.error("System busy. Please try again in a moment.")
                else:
                    message_placeholder.error(f"An error occurred: {e}")
    else:
        st.error(f"System Error: No documents loaded for {selected_state_name}.")

# Sidebar Info
with st.sidebar:
    st.markdown("### Selected State")
    st.markdown(f"**{selected_state_name} ({selected_state_abbr})**")

    st.markdown("### Loaded Documents")
    if docs_context:
        for f in docs_context:
            st.markdown(f"- {f.display_name}")
    else:
        st.warning("No matching documents found.")
    
    st.markdown("---")
    st.markdown("**Control Panel**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()