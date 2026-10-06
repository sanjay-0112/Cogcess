"""
Cogcess demo app.

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
import os
import time

import streamlit as st  # pyright: ignore[reportMissingImports]

MAX_WORDS = 1000  # keep the demo fast; longer text is cut to this many words
MAX_GRADE_SCALE = 16  # the grade ruler runs from 0 to this value

SAMPLES = {
    "Dense policy text": (
        "Notwithstanding the aforementioned provisions, the claimant shall, within thirty "
        "days of receipt of the determination, submit documentary evidence substantiating "
        "any contention that the assessment was erroneously computed, failing which the "
        "determination shall be deemed final and binding upon all parties concerned."
    ),
    "Plain-language version": (
        "If you think we got the amount wrong, tell us within 30 days of getting our "
        "decision. Send us papers that show why. If you do not, our decision stays as it "
        "is and you cannot change it later."
    ),
}


# ------------------------------------------------------------
# Load the existing Cogcess pipeline once
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cogcess():
    sys.path.insert(0, "text_branch")
    with contextlib.redirect_stdout(io.StringIO()):
        from text_branch import inference_final
    return inference_final



def get_gemini_client():
    """Create a fast Gemini client with automatic SDK retries disabled."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Cloud "
            "Secrets or set it as an environment variable."
        )

    from google import genai  # pyright: ignore[reportMissingImports]
    from google.genai import types  # pyright: ignore[reportMissingImports]

    # Keep transient failures from causing the SDK to sit through several
    # long automatic retries. The simplify feature handles one short retry
    # itself below.
    retry_options = types.HttpRetryOptions(
        attempts=1,
        http_status_codes=[429, 500, 502, 503, 504],
    )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=retry_options),
    )


def simplify_text(text, level="Moderate"):
    """Rewrite text for cognitive accessibility while preserving its meaning."""
    client = get_gemini_client()

    level_instructions = {
        "Mild": (
            "Make small changes only. Replace unnecessarily difficult words and "
            "slightly simplify sentence structure while staying close to the original."
        ),
        "Moderate": (
            "Use simpler vocabulary and shorter, clearer sentences. Break long sentences "
            "into smaller ones when useful. Preserve all important information."
        ),
        "High": (
            "Use very simple everyday vocabulary, short sentences, clear structure, and "
            "direct wording. Break complex ideas into small steps. Do not remove important facts."
        ),
    }[level]

    prompt = f"""You are Cogcess, an AI assistant for cognitive accessibility.

Rewrite the following text so it is easier for a person with cognitive reading difficulties to understand.

Simplification level: {level}
Instructions: {level_instructions}

Strict rules:
- Preserve the original meaning, facts, numbers, names, dates, and important details.
- Do not invent information or add explanations that are not present in the source.
- Prefer common, concrete words over rare or technical words when the meaning allows it.
- Keep sentences short and direct.
- Use bullet points or short sections when that makes the information easier to follow.
- Preserve headings and the overall logical order when possible.
- Do not mention that you are simplifying the text.
- Return ONLY the rewritten text, with no preamble or commentary.

TEXT TO SIMPLIFY:
{text}
"""

    from google.genai import types  # pyright: ignore[reportMissingImports]

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=4096,
    )

    # Flash-Lite is designed for fast, cost-efficient high-volume tasks.
    # One immediate retry handles a temporary 503/429 without making the
    # user wait through the SDK's longer automatic retry sequence.
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=config,
            )
            simplified = (response.text or "").strip()
            if not simplified:
                raise RuntimeError("Gemini returned an empty response.")
            return simplified
        except Exception as exc:
            error_text = str(exc)
            is_transient = any(
                marker in error_text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
            )
            if attempt == 0 and is_transient:
                time.sleep(1.0)
                continue
            raise

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


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
BAND_COLORS = {
    "Easy": ("#0F7B4F", "#E3F4EC"),
    "Medium": ("#A15C07", "#FCEFD6"),
    "Hard": ("#B42318", "#FBE4E1"),
    "Unknown": ("#6F675A", "#ECE4D3"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

:root {
  --ink: #2B2A26;
  --muted: #6F675A;
  --paper: #F2EADB;
  --surface: #FBF7EE;
  --line: #DCCFB8;
  --accent: #0E6B73;
  --accent-soft: #DCE9E4;
  --line-strong: #C9BC9F;
  --sidebar: #EDE3CF;
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
}
.stApp { background: var(--paper); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }

h1, h2, h3, .serif { font-family: 'Newsreader', Georgia, serif; letter-spacing: -0.01em; }

/* Masthead */
.masthead { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.logo {
  width: 44px; height: 44px; border-radius: 12px; background: var(--ink);
  display: flex; align-items: center; justify-content: center; font-size: 22px;
}
.brand { font-family: 'Newsreader', serif; font-size: 2rem; font-weight: 600; line-height: 1; }
.tagline { color: var(--muted); font-size: 1.02rem; max-width: 62ch; line-height: 1.55; margin: 6px 0 22px; }

/* Panels */
.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 24px; margin-bottom: 14px;
}
.panel-title { font-weight: 600; font-size: 0.95rem; color: var(--muted); margin-bottom: 10px; }

/* Verdict */
.verdict { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
.grade-num { font-family: 'Newsreader', serif; font-size: 4.4rem; font-weight: 500; line-height: 0.95; }
.grade-label { color: var(--muted); font-size: 0.92rem; margin-top: 6px; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; }
.chip { border-radius: 999px; padding: 7px 16px; font-weight: 600; font-size: 0.92rem; }
.chip.outline { border: 1px solid var(--line); background: var(--surface); color: var(--ink); }

/* Grade ruler */
.ruler { position: relative; margin: 28px 0 8px; height: 10px; border-radius: 999px;
  background: linear-gradient(90deg, #0F7B4F 0%, #0F7B4F 40%, #D9961A 44%, #D9961A 68%, #B42318 72%, #B42318 100%); }
.ruler-pin { position: absolute; top: -7px; width: 4px; height: 24px; border-radius: 2px; background: var(--ink);
  box-shadow: 0 0 0 3px var(--surface); transform: translateX(-2px); }
.ruler-scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.8rem; margin-top: 10px; }

/* Meters */
.meter-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.meter-name { font-weight: 500; }
.meter-val { font-family: 'Newsreader', serif; font-size: 1.6rem; font-weight: 500; }
.meter-track { height: 8px; border-radius: 999px; background: #E6DCC8; overflow: hidden; }
.meter-fill { height: 100%; border-radius: 999px; background: var(--accent); }
.meter-note { color: var(--muted); font-size: 0.85rem; margin-top: 8px; line-height: 1.45; }

/* Change banner */
.change { border-radius: 14px; padding: 18px 22px; display: flex; align-items: center; gap: 18px; margin-top: 6px; }
.change-num { font-family: 'Newsreader', serif; font-size: 2.2rem; font-weight: 600; }
.change-text { font-size: 1rem; line-height: 1.45; }

/* Streamlit widgets */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }
.stTextArea textarea {
  border-radius: 12px; border: 1px solid var(--line); background: var(--surface);
  font-family: 'IBM Plex Sans', sans-serif; font-size: 1rem; line-height: 1.6;
}
.stTextArea textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; padding: 0.6rem 1.4rem; border: 1px solid var(--line);
}
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #0B5A61; border-color: #0B5A61; }
[data-testid="stFileUploaderDropzone"] { border-radius: 12px; background: var(--surface); border: 1px dashed var(--line-strong); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
.foot { color: var(--muted); font-size: 0.82rem; text-align: center; margin-top: 40px; line-height: 1.6; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---- Readability fixes: force dark text on light surfaces even if the viewer uses dark mode ---- */
:root, html, body, .stApp, [data-testid="stSidebar"] { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stSidebarContent"] { color: var(--ink); }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4, [data-testid="stSidebar"] li, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] strong, h1, h2, h3, h4 { color: var(--ink) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] p { color: var(--muted) !important; font-weight: 500; }
.stTabs [aria-selected="true"] p { color: var(--ink) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* Inputs */
.stTextArea textarea { background: var(--surface) !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }
.stTextArea textarea::placeholder { color: #7A7060 !important; -webkit-text-fill-color: #7A7060; opacity: 1; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface) !important; border: 1px solid var(--line-strong) !important; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color: var(--ink) !important; }

/* Buttons: light secondary buttons, teal primary button */
.stButton > button[kind="secondary"] { background: var(--surface) !important; border: 1px solid var(--line-strong) !important; }
.stButton > button[kind="secondary"], .stButton > button[kind="secondary"] * { color: var(--ink) !important; }
.stButton > button[kind="secondary"]:hover { border-color: var(--accent) !important; background: var(--accent-soft) !important; }
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * { color: #fff !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; }

/* Expander, code, alerts, spinner */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code { background: #EFE6D4 !important; color: var(--ink) !important; }
[data-testid="stAlert"], [data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--ink) !important; }

/* Spinners and tab underline */
[data-testid="stSpinner"], [data-testid="stSpinner"] *, .stSpinner, .stSpinner * {
  background-color: transparent !important; color: var(--ink) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
.stTabs [data-baseweb="tab-border"] { background-color: var(--line) !important; }
</style>
"""


def meter_html(name, value_text, percent, note):
    percent = max(0.0, min(100.0, percent))
    return (
        '<div class="panel">'
        f'<div class="meter-row"><span class="meter-name">{name}</span>'
        f'<span class="meter-val">{value_text}</span></div>'
        f'<div class="meter-track"><div class="meter-fill" style="width:{percent:.1f}%"></div></div>'
        f'<div class="meter-note">{note}</div>'
        "</div>"
    )


def show_result(result):
    grade = result["grade"]
    level = band(grade)
    fg, bg = BAND_COLORS[level]
    grade_text = f"{grade:.1f}" if grade is not None else "n/a"

    chips = f'<span class="chip" style="color:{fg};background:{bg}">{level} to read</span>'
    if result["cefr"]:
        chips += f'<span class="chip outline">CEFR {result["cefr"]}</span>'

    pin = ""
    if grade is not None:
        pos = max(0.0, min(100.0, grade / MAX_GRADE_SCALE * 100))
        pin = f'<div class="ruler-pin" style="left:{pos:.1f}%"></div>'

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">Predicted reading level</div>'
        '<div class="verdict">'
        f'<div><div class="grade-num">{grade_text}</div>'
        '<div class="grade-label">US school grade needed to follow this text</div></div>'
        f'<div class="chips">{chips}</div>'
        "</div>"
        f'<div class="ruler">{pin}</div>'
        '<div class="ruler-scale"><span>Grade 0</span><span>Grade 8</span><span>Grade 16</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    if result["difficult_ratio"] is not None:
        c1.markdown(
            meter_html(
                "Difficult words",
                f"{result['difficult_ratio']:.1f}%",
                result["difficult_ratio"],
                "Share of words that are likely to slow readers down.",
            ),
            unsafe_allow_html=True,
        )
    if result["lexical_mean"] is not None:
        c2.markdown(
            meter_html(
                "Average word difficulty",
                f"{result['lexical_mean']:.1f} / 100",
                result["lexical_mean"],
                "Higher means rarer or more complex vocabulary.",
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Full Cogcess report"):
        st.code(result["report"])


def show_change(r1, r2):
    if r1["grade"] is None or r2["grade"] is None:
        return
    change = r1["grade"] - r2["grade"]
    if change > 0.05:
        fg, bg, msg = "#0F7B4F", "#E3F4EC", "The simplified text is easier to read."
    elif change < -0.05:
        fg, bg, msg = "#B42318", "#FBE4E1", "The simplified text is harder to read than the original."
    else:
        fg, bg, msg = "#6F675A", "#ECE4D3", "The two texts have about the same reading level."
    st.markdown(
        f'<div class="change" style="background:{bg};color:{fg}">'
        f'<div class="change-num">{change:+.1f}</div>'
        f'<div class="change-text"><b>Grade level change.</b><br>{msg} '
        "A positive number means simpler.</div></div>",
        unsafe_allow_html=True,
    )


def safe_run(text):
    """Run the pipeline and show a clear error instead of a stack trace."""
    try:
        return run_cogcess(text)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Cogcess could not analyze this text. Check that the models are in place "
            f"and try again. Details: {exc}"
        )
        return None


def set_sample(key, text):
    st.session_state[key] = text


def word_count_caption(text):
    n = len(text.split())
    note = f"{n:,} words"
    if n > MAX_WORDS:
        note += f" (only the first {MAX_WORDS:,} will be analyzed)"
    st.caption(note)


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
st.set_page_config(page_title="Cogcess", page_icon="🧠", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand" style="font-size:1.5rem">Cogcess</div>', unsafe_allow_html=True)
    st.caption("Cognitive accessibility toolkit")
    st.markdown("#### How to read the results")
    st.markdown(
        "- **Grade level** estimates the US school grade needed to follow the text.\n"
        "- **CEFR** places the language on a scale from A1 (beginner) to C2 (mastery).\n"
        "- **Difficult words** shows how much of the text uses harder vocabulary."
    )
    st.markdown("#### Try an example")
    st.button("Load dense policy text", on_click=set_sample, args=("typed", SAMPLES["Dense policy text"]), use_container_width=True)
    st.button("Load plain-language text", on_click=set_sample, args=("typed", SAMPLES["Plain-language version"]), use_container_width=True)
    st.markdown("#### Limits")
    st.caption(
        f"Texts longer than {MAX_WORDS:,} words are cut to keep the demo fast. "
        "Easy, Medium and Hard bands are demo thresholds and are not scientifically validated."
    )

st.markdown(
    '<div class="masthead"><div class="logo">🧠</div><div class="brand">Cogcess</div></div>'
    '<div class="tagline">See how hard your text is to read, and check whether a rewrite actually helps. '
    "Readability is one part of cognitive accessibility, not the whole picture. "
    "Dense or complex text can be harder for some people to follow.</div>",
    unsafe_allow_html=True,
)

tab_one, tab_simplify, tab_compare = st.tabs(["Analyze text", "Simplify text", "Compare two texts"])

with tab_one:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a .txt or .pdf file (optional)", type=["txt", "pdf"])
    typed = st.text_area("Or paste text here", height=220, key="typed", placeholder="Paste the text you want to check...")
    word_count_caption(typed)

    if st.button("Analyze text", type="primary"):
        try:
            text = read_upload(uploaded) if uploaded is not None else typed
        except Exception as exc:  # noqa: BLE001
            st.error(f"That file could not be read. Try a different file. Details: {exc}")
            text = ""
        text, cut = clean_text(text)
        if not text.strip():
            st.warning("Paste some text or upload a file to analyze.")
        else:
            if cut:
                st.info(f"Your text is long, so only the first {MAX_WORDS:,} words were analyzed.")
            with st.spinner("Analyzing. The first run loads the models and can take a while..."):
                result = safe_run(text)
            if result:
                show_result(result)

with tab_simplify:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.write("Rewrite difficult text into a clearer, more accessible version using AI while preserving its meaning.")

    uploaded_s = st.file_uploader("Upload a .txt or .pdf file (optional)", type=["txt", "pdf"], key="simplify_upload")
    typed_s = st.text_area(
        "Or paste text here",
        height=220,
        key="simplify_typed",
        placeholder="Paste the text you want Cogcess to simplify...",
    )
    level = st.selectbox(
        "Simplification level",
        ["Mild", "Moderate", "High"],
        index=1,
        help="Mild stays close to the original. High uses shorter sentences and simpler vocabulary.",
    )
    word_count_caption(typed_s)

    if st.button("Simplify with Cogcess", type="primary", key="simplify_button"):
        try:
            text = read_upload(uploaded_s) if uploaded_s is not None else typed_s
        except Exception as exc:  # noqa: BLE001
            st.error(f"That file could not be read. Try a different file. Details: {exc}")
            text = ""

        text, cut = clean_text(text)
        if not text.strip():
            st.warning("Paste some text or upload a file to simplify.")
        else:
            if cut:
                st.info(f"Your text is long, so only the first {MAX_WORDS:,} words were sent for simplification.")
            try:
                with st.spinner("Simplifying with Cogcess AI..."):
                    simplified = simplify_text(text, level)
                st.success("Simplified version generated.")

                left, right = st.columns(2, gap="large")
                with left:
                    st.subheader("Original")
                    st.markdown(text)
                with right:
                    st.subheader("Cogcess simplified")
                    st.markdown(simplified)

                st.download_button(
                    "Download simplified text",
                    data=simplified,
                    file_name="cogcess_simplified.txt",
                    mime="text/plain",
                )

                st.markdown("### Readability improvement")
                with st.spinner("Checking the simplified version with the Cogcess analysis engine..."):
                    original_result = safe_run(text)
                    simplified_result = safe_run(clean_text(simplified)[0])
                if original_result and simplified_result:
                    show_change(original_result, simplified_result)
                    left, right = st.columns(2, gap="large")
                    with left:
                        st.subheader("Before")
                        show_result(original_result)
                    with right:
                        st.subheader("After")
                        show_result(simplified_result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Simplification failed. {exc}")

with tab_compare:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.write("Paste an original text and a simplified version to see how the reading level changes.")
    left, right = st.columns(2, gap="large")
    original = left.text_area("Original", height=220, key="orig")
    simplified = right.text_area("Simplified", height=220, key="simp")

    if st.button("Compare texts", type="primary"):
        if not original.strip() or not simplified.strip():
            st.warning("Fill in both boxes to compare.")
        else:
            with st.spinner("Analyzing both texts. The first run loads the models and can take a while..."):
                r1 = safe_run(clean_text(original)[0])
                r2 = safe_run(clean_text(simplified)[0]) if r1 else None
            if r1 and r2:
                show_change(r1, r2)
                left, right = st.columns(2, gap="large")
                with left:
                    st.subheader("Original")
                    show_result(r1)
                with right:
                    st.subheader("Simplified")
                    show_result(r2)

st.markdown(
    '<div class="foot">Cogcess is a research demo. Scores are estimates to guide editing, '
    "not a verdict on any reader's ability.</div>",
    unsafe_allow_html=True,
)
