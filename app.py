import streamlit as st
import os
import glob
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

# Initialize the Client (Gemini 3 requires this new client)
client = genai.Client(api_key=api_key)

# 2. Page Configuration
st.set_page_config(page_title="Medicaid Benefits Navigation", page_icon="🛑", layout="centered")

# 3. Custom CSS for AARP Styling & Visibility Fixes
st.markdown("""
    <style>
    /* 1. Global Reset to ensure Light Mode colors even if user system is Dark */
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    
    /* 2. Text Colors - Force Dark Gray globally */
    p, div, h1, h2, h3, h4, h5, h6, span, label, li {
        color: #333333 !important;
    }

    /* 3. Sidebar Specifics - Fix for "White Text on White BG" */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4 !important; /* Light Gray Background */
        border-right: 2px solid #E60000;
    }
    
    /* Force sidebar text to be black */
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* 4. AARP Red Highlights */
    h1, h2, h3 {
        color: #E60000 !important; /* AARP Red Headers */
    }
    
    /* Custom Text Selection (When you highlight text with mouse) */
    ::selection {
        background: #E60000; /* Red Background */
        color: #FFFFFF !important; /* White Text */
    }

    /* 5. Chat Bubbles */
    div[data-testid="stChatMessageContent"] {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 8px;
    }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #E60000;
        color: white !important;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #cc0000; /* Darker red on hover */
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    # Use the robot emoji as placeholder
    st.markdown("<div style='font-size: 40px;'>🤖</div>", unsafe_allow_html=True) 
with col2:
    st.title("Benefits Navigator Assistant")
    st.markdown(f"**AARP Foundation | Pennsylvania Pilot**")

st.divider()

# 5. Multimodal File Handling
@st.cache_resource
def load_docs_to_gemini():
    """
    Scans 'docs/' folder and uploads PDFs.
    """
    uploaded_files = []
    doc_paths = glob.glob("docs/*.pdf")

    if not doc_paths:
        st.sidebar.error("No PDF files found in 'docs/' folder.")
        return []

    status_bar = st.sidebar.status("System Status: Loading Policy Documents...")
    
    for path in doc_paths:
        file_name = os.path.basename(path)
        status_bar.write(f"Indexing: {file_name}")
        
        try:
            with open(path, "rb") as f:
                sample_file = client.files.upload(
                    file=f, 
                    config={
                        'display_name': file_name,
                        'mime_type': 'application/pdf' # Strict MIME type check
                    }
                )
            uploaded_files.append(sample_file)
        except Exception as e:
            st.sidebar.error(f"Failed to load {file_name}: {e}")
            
    status_bar.update(label="System Ready", state="complete", expanded=False)
    return uploaded_files

# Load the files
docs_context = load_docs_to_gemini()

# 6. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. The Chat Loop
if prompt := st.chat_input("Enter eligibility question here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    if docs_context:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing policy documents...")
            
            try:
                # Prepare content list
                content_payload = [
                    "You are a strict Medicaid Eligibility Assistant. Use the attached policy documents to answer.",
                    "Cite the specific document name or section when possible.",
                    "If the answer depends on a checked box, explicitly state that.",
                ]
                content_payload.extend(docs_context)
                content_payload.append(prompt)

                # Call Gemini 3 Flash Preview
                response = client.models.generate_content(
                    model="gemini-2.0-flash", # Use 2.0 Flash or 1.5 Flash for stability if 3-preview is glitchy
                    contents=content_payload
                )
                
                # Display result
                message_placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")
    else:
        st.error("System Error: No documents loaded. Please add PDFs to the 'docs' folder.")

# Sidebar Info
with st.sidebar:
    st.markdown("### Loaded Documents")
    if docs_context:
        for f in docs_context:
            st.markdown(f"📄 **{f.display_name}**") # Bold for better visibility
    else:
        st.warning("No documents found.")
    
    st.markdown("---")
    st.markdown("**Control Panel**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()