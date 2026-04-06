# AARP Benefits Navigator

A multimodal AI assistant for navigating complex Medicaid policy documents. 

This system moves beyond basic text extraction by utilizing a **Layout Aware Retrieval Augmented Generation** pipeline. It uses LlamaParse to understand visual PDF structures (like dense tables and eligibility checkboxes) and the Gemini model to generate precise, citation backed answers for caregivers.

This guide serves as a quick start handover for collaborators and stakeholders.

---

## 1. Prerequisites

You will need two free API keys to run the visual parsing and reasoning models:

* **Python 3.9 or higher**
* **Google Gemini API Key**: Obtain a free key from Google AI Studio.
* **LlamaCloud API Key**: Obtain a free key from LlamaIndex Cloud for the visual parsing engine.

*Note: If you face errors obtaining a key, try using a personal email address, as some enterprise organizations restrict API generation.*

---

## 2. Installation

### Clone the Repository

```bash
git clone https://github.com/moidulhn/aarp.git
cd aarp_pilot_demo
```

### Create a Virtual Environment

#### macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configuration

### Environment Variables

Create a file named `.env` in the project root directory. Add both of your API keys:

```plaintext
GOOGLE_API_KEY="your_google_key_here"
LLAMA_CLOUD_API_KEY="your_llama_cloud_key_here"
```

### Policy Documents

Place your PDF policy manuals (for example CHC Waiver and OBRA Waiver) inside the `docs/` folder.

---

## 4. Running the Application

This enterprise architecture uses a two step process. You must build the structural index first before launching the chat interface.

### Step 1: Build the Knowledge Base

Run the ingestion pipeline. This script sends the PDFs to the LlamaCloud vision agent, extracts the visual layouts into Markdown, and builds a local FAISS vector database in the `storage/` folder.

```bash
python pipeline.py
```

You only need to run this command once, or whenever updated PDFs are added.

---

### Step 2: Launch the Interface

Once the terminal confirms the knowledge base is built, launch the Streamlit chat application:

```bash
streamlit run app.py
```

The Benefits Navigator will open in your default web browser.

---

## 5. System Features

### Layout Aware Parsing

Uses the Agentic tier of LlamaParse to read documents like a human.

Preserves complex table relationships.

Identifies the state of visual checkboxes (Checked vs Unchecked).

---

### Semantic Chunking and Local Search

Splits documents by logical headers rather than arbitrary chunk sizes.

Uses FAISS for fast local vector retrieval.

Reduces token usage compared to full document processing.

---

### Citation Grounding

The Gemini model is strictly prompted to use retrieved context only.

Every answer includes a reference to the exact source section used.

---

## 6. Project Structure

```plaintext
aarp_pilot_demo/
│
├── app.py
├── pipeline.py
├── requirements.txt
├── .env
├── docs/
├── storage/
└── README.md
```

---

## Key Upgrades in this Version

* Clear separation between ingestion pipeline and application interface
* Explicit support for layout aware document understanding
* Improved handling of structured data such as tables and checkboxes
* Scalable architecture using FAISS based retrieval