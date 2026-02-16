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

# 2. Page Configuration
st.set_page_config(page_title="Medicaid Benefits Navigation", layout="centered")

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
    </style>
""", unsafe_allow_html=True)

# 4. Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown(f"**AARP Foundation | Pennsylvania Pilot**")

st.divider()

# 5. Multimodal File Handling
@st.cache_resource
def load_docs_to_gemini():
    uploaded_files = []
    doc_paths = glob.glob("docs/*.pdf")
    if not doc_paths:
        st.sidebar.error("No PDF files found in 'docs/' folder.")
        return []
    status_bar = st.sidebar.status("System Status: Connecting to Gemini...")
    existing_files_map = {}
    try:
        for f in client.files.list():
            existing_files_map[f.display_name] = f
    except Exception as e:
        st.sidebar.warning(f"Could not list existing files: {e}")
    for path in doc_paths:
        file_name = os.path.basename(path)
        try:
            if file_name in existing_files_map:
                uploaded_files.append(existing_files_map[file_name])
            else:
                with open(path, "rb") as f:
                    sample_file = client.files.upload(
                        file=f, 
                        config={'display_name': file_name, 'mime_type': 'application/pdf'}
                    )
                uploaded_files.append(sample_file)
                time.sleep(1)
        except Exception as e:
            st.sidebar.error(f"Failed to load {file_name}: {e}")
    status_bar.update(label="System Ready", state="complete", expanded=False)
    return uploaded_files

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
                # ENHANCED SYSTEM PROMPT FOR CITATIONS AND VISUAL LOGIC
                content_payload = [
                    "You are a strict Medicaid Eligibility Assistant. You must use the provided documents only.",
                    "STRICT RULE: For every fact or eligibility requirement you mention, you MUST provide the document name and page number.",
                    "FORMAT: Use bold citations like **[Document Name, Page XX]** at the end of relevant sentences or paragraphs.",
                    "VISUAL CHECK: If a checkbox is marked (X or check), state 'The box for [Title] is visually checked on page [XX]'.",
                    "If you are unsure or the information is not present, state that you do not know or are unsure.",
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
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=content_payload)
                    message_placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
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
    st.markdown("---")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()