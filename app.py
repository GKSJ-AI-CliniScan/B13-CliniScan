import streamlit as st
from PIL import Image
from utils.predict import run_models
from utils.report import generate_pdf_report
from streamlit_lottie import st_lottie
import json
import os

# ---------------------------
# LOAD CSS
# ---------------------------
def load_css():
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------
# LOAD LOTTIE
# ---------------------------
def load_lottie(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

lottie_scan = load_lottie("assets/Scan_Animation.json")

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="CliniScan AI",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------
# SESSION STATE
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------------------
# NAVBAR (FIXED 🔥)
# ---------------------------
col1, col2 = st.columns([2, 6])

with col1:
    st.markdown("### 🩺 CliniScan AI")

with col2:
    nav_cols = st.columns(4)

    def nav_button(label, page_name):
        if st.session_state.page == page_name:
            st.markdown(
                f"<div class='nav-btn active'>{label}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.button(label, key=page_name):
                st.session_state.page = page_name

    with nav_cols[0]:
        nav_button("Home", "Home")
    with nav_cols[1]:
        nav_button("Analyze", "Analyze")
    with nav_cols[2]:
        nav_button("Features", "Features")
    with nav_cols[3]:
        nav_button("Tech", "Tech Stack")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# HOME PAGE
# ---------------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="badge">YOLOv8 + EfficientNet + Grad-CAM</div>
        <h1>AI-Powered Chest X-ray Diagnosis</h1>
        <p>Upload X-rays and get instant AI-driven clinical insights</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Classification", "15 Classes")
    col2.metric("Detection", "14 Classes")
    col3.metric("Explainability", "Grad-CAM")

# ---------------------------
# ANALYZE PAGE
# ---------------------------
if st.session_state.page == "Analyze":

    col1, col2 = st.columns([1, 2])

    with col1:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Upload X-ray")

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg"]
        )

        analyze_btn = st.button("Analyze")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

            if analyze_btn:

                loading_placeholder = st.empty()

                with loading_placeholder.container():
                    st.markdown("<div class='card loading-box'>", unsafe_allow_html=True)
                    st.markdown("### AI is analyzing the scan...")

                    if lottie_scan:
                        st_lottie(lottie_scan, height=220)
                    else:
                        st.info("Analyzing X-ray...")

                    st.markdown("</div>", unsafe_allow_html=True)

                result = run_models(image)

                loading_placeholder.empty()
                st.success("Analysis Complete")

                # st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Diagnosis")

                st.markdown(f"## {result['classification']}")
                st.progress(float(result["confidence"]))
                st.write(f"Confidence: {result['confidence']:.2f}")

                if result["classification"] != "No Finding":
                    st.error("Abnormality Detected")
                else:
                    st.success("Normal Scan")

                st.info(result["note"])
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### Visual Analysis")

                v1, v2 = st.columns(2)

                with v1:
                    # st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Detection (YOLO)")
                    st.image(result["detection_image"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with v2:
                    # st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Grad-CAM")
                    st.image(result["gradcam_image"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Clinical Interpretation")

                if result["classification"] != "No Finding":
                    st.write("Potential abnormality detected. Clinical correlation recommended.")
                else:
                    st.write("No significant abnormalities detected.")

                st.markdown("</div>", unsafe_allow_html=True)

                # st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Download Report")

                pdf_path = generate_pdf_report(
                    result, image,
                    result["detection_image"],
                    result["gradcam_image"]
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f,
                        file_name="CliniScan_Report.pdf"
                    )

                st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# FEATURES
# ---------------------------
if st.session_state.page == "Features":
    st.markdown("## Features")

    st.markdown("""
    <div class='card'>
    Deep Learning Classification<br>
    YOLOv8 Object Detection<br>
    Grad-CAM Explainability<br>
    Automated PDF Reports<br>
    Fast AI Inference
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# TECH STACK
# ---------------------------
if st.session_state.page == "Tech Stack":
    st.markdown("## Tech Stack")

    st.markdown("""
    <div class='card'>
    Python<br>
    PyTorch<br>
    YOLOv8<br>
    Streamlit<br>
    ReportLab
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>⚕️ CliniScan AI • AI Medical Assistant</p>",
    unsafe_allow_html=True
)