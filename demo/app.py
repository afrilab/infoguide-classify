"""InfoGuide demo dashboard — entrypoint.

Run from the project root:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

# Make `demo/` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="InfoGuide Demo",
    page_icon=None,
    layout="wide",
)

st.title("InfoGuide — Pipeline Inspection Dashboard")

st.markdown(
    """
This dashboard inspects **precomputed** outputs of the InfoGuide pipeline
(ingestion → preprocessing → anonymization → classification → taxonomy).
Use the sidebar to navigate.

**Pages**
- **Browse** — pick a corpus + document and view the raw content.
- **Pipeline View** — for one document, inspect every stage's output side-by-side
  with selectable model / method variants per module.
- **Classification Evaluation Outputs** — review aggregate classification reports,
  model comparisons, and document-level evaluation CSVs.
- **Anonymization Evaluation Outputs** — review aggregate anonymization metrics,
  breakdowns, and error rows.
- **Design Evaluations** — per-module design and evaluation reports (narrative
  markdown documents on rationale, trade-offs, and findings).
"""
)

st.info(
    "Tip: the document selected on the **Browse** page is shared across all pages "
    "via session state."
)
