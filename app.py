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

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    # Professional Text Placeholder (No Robot Emoji)
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown(f"**AARP Foundation | Pennsylvania Pilot**")

st.divider()

# 5. Multimodal File Handling (Smart Version - Prevents "Stuck" Spinner)
@st.cache_resource
def load_docs_to_gemini():
    """
    Scans 'docs/' folder. Checks if files are already uploaded to Gemini 
    to avoid re-uploading and hitting rate limits.
    """
    uploaded_files = []
    doc_paths = glob.glob("docs/*.pdf")

    if not doc_paths:
        st.sidebar.error("No PDF files found in 'docs/' folder.")
        return []

    status_bar = st.sidebar.status("System Status: Connecting to Gemini...")
    
    # Step 1: List files already on Google's server
    # We create a dictionary map: {'filename.pdf': file_object}
    existing_files_map = {}
    try:
        # We list existing files to see what is already there
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
                # File exists! Skip upload and just use it.
                status_bar.write(f"✅ Found cached: {file_name}")
                uploaded_files.append(existing_files_map[file_name])
            else:
                # File is new! Upload it.
                status_bar.write(f"⬆️ Uploading: {file_name}...")
                with open(path, "rb") as f:
                    sample_file = client.files.upload(
                        file=f, 
                        config={
                            'display_name': file_name,
                            'mime_type': 'application/pdf' # Crucial for Gemini 3
                        }
                    )
                uploaded_files.append(sample_file)
                # Sleep briefly to be kind to the API rate limit
                time.sleep(1)
                
        except Exception as e:
            st.sidebar.error(f"Failed to load {file_name}: {e}")
            
    status_bar.update(label="System Ready", state="complete", expanded=False)
    return uploaded_files

# Load the files
docs_context = load_docs_to_gemini()

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
                # Prepare content
                content_payload = [
                    "You are a strict Medicaid Eligibility Assistant. Use the attached policy documents to answer.",
                    "Cite the specific document name or section when possible.",
                    "If the answer depends on a checked box, explicitly state that.",
                ]
                content_payload.extend(docs_context)
                content_payload.append(prompt)

                # Generate Content 
                # Using Gemini 3 Flash Preview as requested
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", 
                    contents=content_payload
                )
                
                message_placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                # Graceful Error Handling
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
                    except:
                        message_placeholder.error("System busy. Please try again in a moment.")
                else:
                    message_placeholder.error(f"An error occurred: {e}")
    else:
        st.error("System Error: No documents loaded.")

# Sidebar Info
with st.sidebar:
    st.markdown("### Loaded Documents")
    if docs_context:
        for f in docs_context:
            st.markdown(f"- {f.display_name}")
    else:
        st.warning("No documents found.")
    
    st.markdown("---")
    st.markdown("**Control Panel**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()