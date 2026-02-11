````markdown
# AARP Benefits Navigator

A multimodal AI assistant for navigating Medicaid policy documents, starting with the state of Pennsylvania.  
Powered by **Gemini 3.0 Flash Preview**, the system analyzes text, tables, and form elements (including checkboxes) within PDF files.

This README serves as a quick-start handover guide for collaborators. The application should be runnable in under five minutes.

---

## 1. Prerequisites

- **Python 3.9 or higher**
- **Google Gemini API Key**  
  Obtain a free API key from:  
  https://aistudio.google.com/app/apikey

---

## 2. Installation

### Clone the Repository

```bash
git clone https://github.com/moidulhn/aarp.git
cd aarp-pilot-demo
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

Create a file named `.env` in the project root directory:

```text
GOOGLE_API_KEY=your_actual_key_here
```

Replace `your_actual_key_here` with your actual Gemini API key. Ensure that the key is enclosed within single quotes.

---

### Policy Documents

Place any PDF policy manuals (e.g., *Appendix B*) inside the:

```
docs/
```

folder.

On startup, the application automatically scans and indexes all PDFs in this folder.

---

## 4. Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The app will open in your default browser.

---

## System Features

### Smart Indexing

- Checks whether documents are already uploaded to Gemini cloud storage
- Prevents redundant uploads
- Reduces rate limit exposure

### Multimodal Document Understanding

- Uses the **Google GenAI SDK**
- Extracts structured information from:
  - Tables
  - Checkboxes
  - Complex PDF structures

---

## Project Structure

```text
aarp-pilot-demo/
│
├── app.py
├── requirements.txt
├── .env
├── docs/
│   └── (PDF policy documents)
└── README.md
```

---
````