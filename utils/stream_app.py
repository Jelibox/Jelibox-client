"""
stream_app.py  —  Jelibox · Model Demo
Live inference viewer powered by Streamlit + Ultralytics.

Usage (launched automatically by AnnotationGUI):
    streamlit run stream_app.py -- <model_path>
"""

import sys
import time
import tempfile
import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# ──────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Model Demo · Jelibox",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  Theme + CSS  (palette shared with the desktop app via utils/theme.py)
# ──────────────────────────────────────────────
import base64

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _ROOT)
from utils import theme as T  # noqa: E402

try:
    with open(os.path.join(_ROOT, "assets", "jelibox.png"), "rb") as _f:
        _LOGO = "data:image/png;base64," + base64.b64encode(_f.read()).decode()
except OSError:
    _LOGO = ""

_CSS = """
<style>
:root { --bg:@BASE@; --panel:@PANEL@; --card:@CARD@; --card2:@CARD2@; --line:@BORDER@; --accent:@ACCENT@;
        --on-accent:@ON_ACCENT@; --txt1:@TXT1@; --txt2:@TXT2@; --txt3:@TXT3@; --green:@GREEN@; }
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--txt1); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { background: var(--panel) !important; }
/* ── header bar ── */
.bx-header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 14px 32px; margin: -1rem -1rem 0 -1rem; display: flex; align-items: center; gap: 14px; }
.bx-logo { width: 38px; height: 38px; border-radius: 10px; }
.bx-title { font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; color: var(--txt1); line-height: 1.1; }
.bx-subtitle { font-size: .72rem; color: var(--txt2); letter-spacing: .08em; text-transform: uppercase; margin-top: 3px; }
/* ── status badge ── */
.bx-badge { display: inline-block; padding: 3px 14px; border-radius: 999px; font-size: .72rem; font-weight: 700; letter-spacing: .04em; }
.bx-badge-live   { background: var(--card); color: var(--green); border: 1px solid var(--green); }
.bx-badge-idle   { background: var(--card); color: var(--txt2); border: 1px solid var(--line); }
.bx-badge-model  { background: var(--card2); color: var(--txt1); border: 1px solid var(--line); }
/* ── stat cards ── */
.bx-stats { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.bx-stat { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 10px 20px; text-align: center; min-width: 100px; }
.bx-stat-val { font-size: 1.4rem; font-weight: 800; color: var(--accent); }
.bx-stat-lbl { font-size: .68rem; color: var(--txt2); text-transform: uppercase; letter-spacing: .08em; }
/* ── uploader card ── */
.bx-upload-card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px 14px 20px; margin-bottom: 16px; }
.bx-upload-label { font-size: .78rem; font-weight: 700; color: var(--txt2); text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px; }
/* ── Streamlit widget overrides ── */
section[data-testid="stFileUploadDropzone"] { background: var(--card) !important; border: 1px dashed var(--line) !important; border-radius: 12px !important; color: var(--txt2) !important; }
.stButton > button { background: var(--card2) !important; color: var(--txt1) !important; border: 1px solid var(--line) !important; border-radius: 999px !important; font-weight: 700 !important; }
.stButton > button:hover:not(:disabled) { background: var(--accent) !important; color: var(--on-accent) !important; border-color: var(--accent) !important; }
/* ── frame display ── */
.bx-frame-container { background: var(--bg); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.bx-no-video { background: var(--panel); border: 2px dashed var(--line); border-radius: 16px; height: 420px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--txt3); font-size: 1rem; gap: 12px; }
.bx-no-video-icon { font-size: 3.5rem; }
hr { border-color: var(--line) !important; }
</style>
"""
for _k, _v in dict(BASE=T.C_BASE, PANEL=T.C_PANEL, CARD=T.C_CARD, CARD2=T.C_CARD2, BORDER=T.C_BORDER,
                   ACCENT=T.C_ACCENT, ON_ACCENT=T.C_ON_ACCENT, TXT1=T.C_TXT1, TXT2=T.C_TXT2,
                   TXT3=T.C_TXT3, GREEN=T.C_GREEN).items():
    _CSS = _CSS.replace(f"@{_k}@", _v)
st.markdown(_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Resolve model path  (passed via sys.argv)
# ──────────────────────────────────────────────
_raw_args = sys.argv[1:]
_model_path: str | None = None
for arg in _raw_args:
    if arg not in ("--", "") and not arg.startswith("--"):
        _model_path = arg
        break


# ──────────────────────────────────────────────
#  Session state defaults
# ──────────────────────────────────────────────
if "streaming"     not in st.session_state: st.session_state.streaming     = False
if "frame_count"   not in st.session_state: st.session_state.frame_count   = 0
if "detect_count"  not in st.session_state: st.session_state.detect_count  = 0
if "fps_display"   not in st.session_state: st.session_state.fps_display   = 0.0
if "loop_count"    not in st.session_state: st.session_state.loop_count    = 0
if "cap"           not in st.session_state: st.session_state.cap           = None
if "target_vid"    not in st.session_state: st.session_state.target_vid    = None


# ──────────────────────────────────────────────
#  Model loader (cached per path)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)


# ──────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────
model_name = os.path.basename(_model_path) if _model_path else "—"

st.markdown(f"""
<div class="bx-header">
    <img class="bx-logo" src="{_LOGO}" alt="Jelibox">
    <div>
        <div class="bx-title">Jelibox · Model Demo</div>
        <div class="bx-subtitle">Live Detection Stream</div>
    </div>
    <div style="margin-left:auto; display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span class="bx-badge bx-badge-model">⚡ {model_name}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


if not _model_path or not os.path.exists(_model_path):
    st.error(f"⚠️  Model not found: `{_model_path or 'none'}`\n\nMake sure you launch this app from the Jelibox **🎥 Stream** button.")
    st.stop()


# ──────────────────────────────────────────────
#  Layout: left col = controls, right col = feed
# ──────────────────────────────────────────────
ctrl_col, feed_col = st.columns([1, 2.6], gap="large")

# ── LEFT COLUMN ───────────────────────────────
with ctrl_col:
    
    tab_file, tab_yt = st.tabs(["📁 Local File", "📺 YouTube"])
    
    with tab_file:
        st.markdown('<div class="bx-upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Drop a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            label_visibility="collapsed",
            key="video_uploader",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab_yt:
        st.markdown('<div class="bx-upload-card" style="padding-bottom: 24px;">', unsafe_allow_html=True)
        yt_url = st.text_input(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="youtube_url"
        )
        if yt_dlp is None:
            st.error("`yt-dlp` are not installed, this feature will not work normally")
        st.markdown('</div>', unsafe_allow_html=True)

    video_source_ready = (uploaded_file is not None) or (yt_url.strip() != "")

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95,
        value=0.40, step=0.05,
        format="%.2f",
    )

    st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

    btn_start, btn_stop = st.columns(2)
    with btn_start:
        start_btn = st.button("▶  Start", use_container_width=True, disabled=not video_source_ready)
    with btn_stop:
        stop_btn  = st.button("⏹  Stop",  use_container_width=True, disabled=not st.session_state.streaming)

    if start_btn and video_source_ready:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        
        st.session_state.streaming    = True
        st.session_state.frame_count  = 0
        st.session_state.detect_count = 0
        st.session_state.loop_count   = 1
        
        target_vid = None
        
        if yt_url.strip() != "":
            with st.spinner("Mengekstrak YouTube Stream..."):
                ydl_opts = {'format': 'best[height<=720][ext=mp4]/best'}
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(yt_url, download=False)
                        target_vid = info_dict['url']
                except Exception as e:
                    st.error(f"Could not load YouTube: {e}")
                    st.session_state.streaming = False
        else:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                target_vid = tmp.name
        
        if target_vid:
            st.session_state.target_vid = target_vid
            st.session_state.cap = cv2.VideoCapture(target_vid)
            st.rerun() 

    if stop_btn:
        st.session_state.streaming = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        
        if st.session_state.target_vid and not st.session_state.target_vid.startswith("http"):
            try:
                os.unlink(st.session_state.target_vid)
            except OSError:
                pass
        st.session_state.target_vid = None
        st.rerun()

    st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)

    # Live stats
    st.markdown("**Live Stats**")
    stat_badge = "bx-badge-live" if st.session_state.streaming else "bx-badge-idle"
    stat_text  = "STREAMING" if st.session_state.streaming else "IDLE"
    st.markdown(f'<span class="bx-badge {stat_badge}">{stat_text}</span>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bx-stats" style="margin-top:10px">
        <div class="bx-stat"><div class="bx-stat-val">{st.session_state.fps_display:.0f}</div><div class="bx-stat-lbl">FPS</div></div>
        <div class="bx-stat"><div class="bx-stat-val">{st.session_state.frame_count}</div><div class="bx-stat-lbl">Frames</div></div>
        <div class="bx-stat"><div class="bx-stat-val">{st.session_state.detect_count}</div><div class="bx-stat-lbl">Detections</div></div>
        <div class="bx-stat"><div class="bx-stat-val">{st.session_state.loop_count}</div><div class="bx-stat-lbl">Loops</div></div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT COLUMN ──────────────────────────────
with feed_col:

    frame_placeholder = st.empty()   
    info_placeholder  = st.empty()   

    if not st.session_state.streaming or st.session_state.cap is None:
        frame_placeholder.markdown("""
        <div class="bx-no-video">
            <div class="bx-no-video-icon">📽️</div>
            <div>Upload a video or enter a YouTube URL, then press <strong>▶ Start</strong></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        model = load_model(_model_path)
        t_loop_start = time.time()
        frames_this_run = 0

        try:
            while st.session_state.streaming:
                
                ret, frame = st.session_state.cap.read()

                if not ret:
                    st.session_state.cap.release()
                    st.session_state.cap = cv2.VideoCapture(st.session_state.target_vid)
                    st.session_state.loop_count += 1
                    continue

                # ── Run YOLO inference ──
                results = model(frame, conf=conf_thresh, verbose=False)
                annotated = results[0].plot()          
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # ── Update counters ──
                st.session_state.frame_count  += 1
                frames_this_run += 1
                n_det = len(results[0].boxes) if results[0].boxes is not None else 0
                st.session_state.detect_count += n_det

                # ── FPS rolling estimate ──
                elapsed = time.time() - t_loop_start
                if elapsed > 0:
                    st.session_state.fps_display = round(frames_this_run / elapsed, 1)

                # ── Render frame ──
                frame_placeholder.image(
                    annotated_rgb,
                    channels="RGB",
                    use_container_width=True,
                    caption=f"Loop #{st.session_state.loop_count} · Frame {st.session_state.frame_count} · {n_det} detection{'s' if n_det != 1 else ''}",
                )

                # ── Detection summary ──
                if results[0].names and n_det > 0:
                    labels = [results[0].names[int(c)] for c in results[0].boxes.cls]
                    counts = {}
                    for lbl in labels:
                        counts[lbl] = counts.get(lbl, 0) + 1
                    summary = "  ·  ".join(f"**{v}× {k}**" for k, v in counts.items())
                    info_placeholder.markdown(f"<div style='font-size:.82rem; color:var(--txt2); margin-top:4px'>🔍 {summary}</div>", unsafe_allow_html=True)
                else:
                    info_placeholder.markdown("<div style='font-size:.82rem; color:var(--txt3); margin-top:4px'>🔍 No detections</div>", unsafe_allow_html=True)

                time.sleep(0.01)

        except Exception as e:
            pass