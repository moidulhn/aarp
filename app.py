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

# 2. Page Configuration (Classic AARP Style)
st.set_page_config(page_title="Medicaid Benefits Navigation", page_icon="🛑", layout="centered")

# 3. Custom CSS for "Old School" AARP Look
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; font-family: 'Arial', sans-serif; color: #333333; }
    h1, h2, h3 { color: #E60000 !important; font-weight: bold; }
    .stChatMessage { border-radius: 5px !important; border: 1px solid #ddd; }
    div[data-testid="stChatMessageContent"] { background-color: #f9f9f9; color: #000; }
    section[data-testid="stSidebar"] { background-color: #f4f4f4; border-right: 2px solid #E60000; }
    div.stButton > button { background-color: #E60000; color: white; border: none; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("🤖", unsafe_allow_html=True) 
with col2:
    st.title("Benefits Navigator Assistant")
    st.markdown(f"**AARP Foundation | Pennsylvania Pilot | Powered by Gemini 3 Flash**")

st.divider()

# 5. Multimodal File Handling (The "Brain")
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
            # Gemini 3 still needs MIME type explicit for PDFs in the SDK
            with open(path, "rb") as f:
                sample_file = client.files.upload(
                    file=f, 
                    config={
                        'display_name': file_name,
                        'mime_type': 'application/pdf'
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
                    model="gemini-3-flash-preview", 
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
            st.text(f"📄 {f.display_name}")
    else:
        st.warning("No documents found.")
    
    st.markdown("---")
    st.markdown("**Control Panel**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()