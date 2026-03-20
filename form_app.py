import streamlit as st
import os
import glob
import time
from dotenv import load_dotenv
from google import genai

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
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    p, div, h1, h2, h3, h4, h5, h6, span, label, li {
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4 !important;
        border-right: 2px solid #E60000;
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    h1, h2, h3 {
        color: #E60000 !important;
    }
    ::selection {
        background: #E60000;
        color: #FFFFFF !important;
    }
    div[data-testid="stChatMessageContent"] {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem;
    }
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
# STATE CONFIG
# -----------------------------
STATE_MAP = {
    "Pennsylvania": "PA",
    "Alabama": "AL",
}

# Minimal county lists for testing
COUNTY_MAP = {
    "Pennsylvania": [
        "Allegheny", "Philadelphia", "Montgomery", "Bucks", "Delaware",
        "Lancaster", "York", "Westmoreland", "Berks", "Erie", "Other"
    ],
    "Alabama": [
        "Jefferson", "Mobile", "Madison", "Montgomery", "Tuscaloosa",
        "Baldwin", "Shelby", "Lee", "Morgan", "Other"
    ]
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
    "Parent",
    "Child",
    "Sibling",
    "Other family member",
    "Friend / non-family caregiver",
    "Other",
]

COVERAGE_OPTIONS = [
    "Medicaid",
    "Medicare",
    "Private insurance",
    "No current coverage",
    "Other",
]

PLACEHOLDER = "-- Select --"

# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 20px; font-weight: bold; color: #E60000;'>[AI]</div>", unsafe_allow_html=True)
with col2:
    st.title("Benefits Navigator")
    st.markdown("**AARP Foundation | Multi-State Pilot**")

st.divider()

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
    medicaid_status = st.radio("Is the person on Medicaid?", ["Yes", "No"])
    medicaid_eligible = None
    if medicaid_status == "No":
        medicaid_eligible = st.radio("If no, are they eligible for Medicaid?", ["Yes", "No", "Unknown"])

    st.markdown("### Location")
    selected_state_name = st.selectbox(
        "State",
        options=list(STATE_MAP.keys()),
        index=0
    )
    # selected_county = st.selectbox(
    #     "County",
    #     options=COUNTY_MAP[selected_state_name]
    # )
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
    lives_with_person = st.radio("Do you live with the person?", ["Yes", "No"])
    same_state = None
    if lives_with_person == "No":
        same_state = st.radio("If no, do you live in the same state as the person?", ["Yes", "No"])

    submitted = st.form_submit_button("Save Intake Information")

if submitted:
    selected_state_abbr = STATE_MAP[selected_state_name]

    all_conditions = selected_conditions.copy()
    if other_condition_checked and other_condition_text.strip():
        all_conditions.append(f"Other: {other_condition_text.strip()}")

    st.session_state.intake_data = {
#        "current_coverage": current_coverage,
        "medicaid_status": medicaid_status,
        "medicaid_eligible": medicaid_eligible,
        "state_name": selected_state_name,
        "state_abbr": selected_state_abbr,
#        "county": selected_county,
        "congregate_setting": congregate_setting,
        "age": age,
        "conditions": all_conditions,
        "relationship": relationship,
        "lives_with_person": lives_with_person,
        "same_state": same_state,
    }
    st.session_state.intake_submitted = True

    # Reset conversation if state changed
    if st.session_state.selected_state_abbr != selected_state_abbr:
        st.session_state.messages = []
        st.session_state.selected_state_abbr = selected_state_abbr

    st.success("Intake information saved.")

# -----------------------------
# DOC LOADING
# -----------------------------
@st.cache_resource
def load_docs_to_gemini(state_abbr):
    """
    Load only PDFs whose filenames begin with the 2-letter state abbreviation.
    Example: PA_waiver1.pdf, AL_familycaregiver.pdf
    """
    uploaded_files = []
    all_doc_paths = glob.glob("docs/*.pdf")

    doc_paths = [
        path for path in all_doc_paths
        if os.path.basename(path)[:2].upper() == state_abbr.upper()
    ]

    if not doc_paths:
        return []

    status_bar = st.sidebar.status(f"System Status: Connecting to Gemini for {state_abbr}...")

    existing_files_map = {}
    try:
        for f in client.files.list():
            existing_files_map[f.display_name] = f
    except Exception as e:
        st.sidebar.warning(f"Could not list existing files: {e}")

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
                    uploaded_file = client.files.upload(
                        file=f,
                        config={
                            "display_name": file_name,
                            "mime_type": "application/pdf"
                        }
                    )
                uploaded_files.append(uploaded_file)
                time.sleep(1)
        except Exception as e:
            st.sidebar.error(f"Failed to load {file_name}: {e}")

    status_bar.update(label=f"System Ready ({state_abbr})", state="complete", expanded=False)
    return uploaded_files

docs_context = []
if st.session_state.intake_submitted:
    docs_context = load_docs_to_gemini(st.session_state.intake_data["state_abbr"])

# -----------------------------
# HELPER: BUILD CASE SUMMARY
# -----------------------------
def build_case_summary(data):
    conditions_text = ", ".join(data["conditions"]) if data["conditions"] else "None reported"

    medicaid_line = f"Is the person on Medicaid? {data['medicaid_status']}."
    if data["medicaid_status"] == "No":
        medicaid_line += f" Medicaid eligibility status: {data['medicaid_eligible']}."

    same_state_line = ""
    if data["lives_with_person"] == "No":
        same_state_line = f" If caregiver does not live with the person, same state status: {data['same_state']}."

    summary = f"""
Case Summary:
- {medicaid_line}
- Location: {data['state_name']})
- Lives in assisted living facility or group home: {data['congregate_setting']}
- Age: {data['age']}
- Conditions: {conditions_text}
- Caregiver relationship: {data['relationship']}
- Caregiver lives with the person: {data['lives_with_person']}.{same_state_line}
""".strip()

    return summary

def build_eligibility_query(data):
    case_summary = build_case_summary(data)

    query = f"""
You are a Medicaid waiver navigation assistant.

Review the attached waiver and policy documents for {data['state_name']} only.

Goal:
Determine whether this person may be eligible for programs, waivers, or pathways that could allow a family caregiver to be paid, or that are relevant to paid family caregiving support.

Instructions:
1. Use only the attached documents.
2. Focus especially on waiver eligibility, caregiver eligibility, participant-directed options, consumer direction, self-direction, attendant care, personal assistance services, and any rules about legally responsible relatives, spouses, guardians, or live-in caregivers being paid.
3. If the person is not clearly eligible, explain what additional facts would be needed.
4. If Medicaid eligibility is a prerequisite, say that clearly.
5. Note any exclusions related to assisted living, group home residence, age, diagnosis category, or caregiver relationship.
6. Cite the specific document name and relevant section or language when possible.
7. Be careful and precise: do not assume eligibility unless the documents support it.

{case_summary}

Please answer in this format:

1. Likely relevant waiver(s) or program(s)
2. Initial assessment of possible eligibility for paid family caregiver arrangements
3. Key reasons
4. Important restrictions or disqualifiers
5. Additional information needed
6. Citations to supporting document(s)
""".strip()

    return query

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
        if docs_context:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Reviewing waiver documents...")

                try:
                    eligibility_prompt = build_eligibility_query(st.session_state.intake_data)

                    content_payload = [eligibility_prompt]
                    content_payload.extend(docs_context)

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=content_payload
                    )

                    message_placeholder.markdown(response.text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.text
                    })

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
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response.text
                            })
                        except Exception:
                            message_placeholder.error("System busy. Please try again in a moment.")
                    else:
                        message_placeholder.error(f"An error occurred: {e}")
        else:
            st.error("No documents loaded for the selected state.")

# -----------------------------
# CHAT HISTORY DISPLAY
# -----------------------------
st.divider()
st.subheader("Ask a Follow-up Question")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# CHAT LOOP
# -----------------------------
if prompt := st.chat_input("Enter eligibility question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if docs_context and st.session_state.intake_submitted:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing policy documents...")

            try:
                case_summary = build_case_summary(st.session_state.intake_data)

                content_payload = [
                    f"You are a strict Medicaid Eligibility Assistant for {st.session_state.intake_data['state_name']} ({st.session_state.intake_data['state_abbr']}).",
                    "Use only the attached policy documents to answer.",
                    "Use the case summary below as the client context.",
                    "Cite the specific document name or section when possible.",
                    "If the answer depends on a checked box, form selection, diagnosis-specific criteria, or additional eligibility facts, explicitly state that.",
                    case_summary,
                    prompt,
                ]
                content_payload.extend(docs_context)

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
        st.error("Please complete the intake form first, and make sure documents are loaded.")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### Selected State")
    if st.session_state.intake_submitted:
        st.markdown(
            f"**{st.session_state.intake_data['state_name']} "
            f"({st.session_state.intake_data['state_abbr']})**"
        )
    else:
        st.markdown("Not selected yet")

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

    if st.button("Clear Intake + Conversation"):
        st.session_state.messages = []
        st.session_state.intake_data = {}
        st.session_state.intake_submitted = False
        st.session_state.selected_state_abbr = None
        st.rerun()