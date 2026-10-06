"""
Cogcess demo app (simple version).

Put this file in the ROOT of the repo (next to the folders
`models/`, `data/`, `text_branch/`) and run:

    streamlit run app.py

It does not change any of your existing code. It calls
text_branch/inference_final.py and shows the result on a web page.
"""

import io
import re
import sys
import contextlib

import streamlit as st  # pyright: ignore[reportMissingImports]

MAX_WORDS = 1000  # keep the demo fast; longer text is cut to this many words


# ------------------------------------------------------------
# Load the existing Cogcess pipeline once
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Cogcess models (first time takes a while)...")
def load_cogcess():
    sys.path.insert(0, "text_branch")
    with contextlib.redirect_stdout(io.StringIO()):
        from text_branch import inference_final
    return inference_final


def run_cogcess(text):
    """Run analyze_text, capture its printed report, pull out key numbers."""
    module = load_cogcess()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module.analyze_text(text)
    report = buffer.getvalue()

    def grab(pattern, cast=float):
        match = re.search(pattern, report)
        return cast(match.group(1)) if match else None

    result = {
        "grade": grab(r"Fused predicted grade:\s*(-?[\d.]+)"),
        "cefr": grab(r"Prediction:\s*([ABC][12])", str),
        "difficult_ratio": grab(r"Difficult word ratio:\s*([\d.]+)%"),
        "lexical_mean": grab(r"Mean difficulty:\s*([\d.]+)/100"),
        "report": report,
    }
    return result


def band(grade):
    # Simple demo thresholds chosen by the team (not scientifically validated).
    if grade is None:
        return "Unknown"
    if grade < 7:
        return "Easy"
    if grade <= 11:
        return "Medium"
    return "Hard"


def clean_text(text):
    words = text.split()
    cut = len(words) > MAX_WORDS
    return " ".join(words[:MAX_WORDS]), cut


def read_upload(uploaded):
    if uploaded is None:
        return ""
    if uploaded.name.lower().endswith(".pdf"):
        from pypdf import PdfReader  # pyright: ignore[reportMissingImports]

        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def show_result(result):
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted grade level", f"{result['grade']:.1f}" if result["grade"] is not None else "n/a")
    col2.metric("Difficulty", band(result["grade"]))
    col3.metric("CEFR level", result["cefr"] or "n/a")

    if result["difficult_ratio"] is not None:
        st.write(f"Difficult words: **{result['difficult_ratio']:.1f}%** of the text")
    if result["lexical_mean"] is not None:
        st.write(f"Average word difficulty: **{result['lexical_mean']:.1f} / 100**")

    with st.expander("Full Cogcess report"):
        st.code(result["report"])


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(page_title="Cogcess", page_icon="🧠")
st.title("🧠 Cogcess")
st.caption(
    "Text readability analyzer. Readability is one part of cognitive accessibility, "
    "not the whole picture. Some people may find dense or complex text harder to follow."
)

tab_one, tab_compare = st.tabs(["Analyze text", "Compare two texts"])

with tab_one:
    uploaded = st.file_uploader("Upload a .txt or .pdf file (optional)", type=["txt", "pdf"])
    typed = st.text_area("...or paste text here", height=200)

    if st.button("Analyze", type="primary"):
        text = read_upload(uploaded) if uploaded is not None else typed
        text, cut = clean_text(text)
        if not text.strip():
            st.warning("Please paste some text or upload a file.")
        else:
            if cut:
                st.info(f"Text was long, so only the first {MAX_WORDS} words were analyzed.")
            with st.spinner("Analyzing..."):
                show_result(run_cogcess(text))

with tab_compare:
    st.write("Paste an original text and a simplified version to see the change.")
    left, right = st.columns(2)
    original = left.text_area("Original", height=200, key="orig")
    simplified = right.text_area("Simplified", height=200, key="simp")

    if st.button("Compare"):
        if not original.strip() or not simplified.strip():
            st.warning("Please fill in both boxes.")
        else:
            with st.spinner("Analyzing both texts..."):
                r1 = run_cogcess(clean_text(original)[0])
                r2 = run_cogcess(clean_text(simplified)[0])
            left.subheader("Original")
            with left:
                show_result(r1)
            right.subheader("Simplified")
            with right:
                show_result(r2)
            if r1["grade"] is not None and r2["grade"] is not None:
                change = r1["grade"] - r2["grade"]
                st.success(f"Grade level changed by {change:+.1f} (positive = simpler).")
