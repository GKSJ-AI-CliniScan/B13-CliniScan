import streamlit as st
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# Page config must be the absolute first Streamlit command
st.set_page_config(
    page_title="CliniScan AI | Chest X-Ray Analysis",
    page_icon="Assets/icon-dark-32x32.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Move model imports below page config
from utils.inference import run_pipeline, device, classifier
from utils.gradcam import generate_gradcam, overlay_gradcam
from utils.report import generate_report

# -------- THEME MANAGEMENT --------
def init_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

init_theme()

# -------- CUSTOM CSS --------
def inject_custom_css():
    is_dark = st.session_state.theme == "dark"
    
    # Theme colors
    if is_dark:
        bg_primary = "#0f172a"
        bg_secondary = "#1e293b"
        bg_card = "#1e293b"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        border_color = "rgba(148, 163, 184, 0.15)"
        accent_color = "#3b82f6"
        accent_hover = "#60a5fa"
        success_bg = "rgba(34, 197, 94, 0.15)"
        success_border = "rgba(34, 197, 94, 0.3)"
        success_text = "#4ade80"
        error_bg = "rgba(239, 68, 68, 0.15)"
        error_border = "rgba(239, 68, 68, 0.3)"
        error_text = "#f87171"
        info_bg = "rgba(59, 130, 246, 0.15)"
        info_border = "rgba(59, 130, 246, 0.3)"
        info_text = "#60a5fa"
        warning_bg = "rgba(234, 179, 8, 0.15)"
        warning_border = "rgba(234, 179, 8, 0.3)"
        warning_text = "#facc15"
        shadow = "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)"
        shadow_hover = "0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3)"
    else:
        bg_primary = "#ffffff"
        bg_secondary = "#f8fafc"
        bg_card = "#ffffff"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        border_color = "rgba(15, 23, 42, 0.08)"
        accent_color = "#2563eb"
        accent_hover = "#1d4ed8"
        success_bg = "rgba(34, 197, 94, 0.1)"
        success_border = "rgba(34, 197, 94, 0.25)"
        success_text = "#16a34a"
        error_bg = "rgba(239, 68, 68, 0.1)"
        error_border = "rgba(239, 68, 68, 0.25)"
        error_text = "#dc2626"
        info_bg = "rgba(59, 130, 246, 0.1)"
        info_border = "rgba(59, 130, 246, 0.25)"
        info_text = "#2563eb"
        warning_bg = "rgba(234, 179, 8, 0.1)"
        warning_border = "rgba(234, 179, 8, 0.25)"
        warning_text = "#ca8a04"
        shadow = "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)"
        shadow_hover = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Reset & Typography */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}
        
        /* Main app background */
        .stApp {{
            background-color: {bg_primary} !important;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {bg_secondary} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }}
        
        h1 {{
            font-size: 2.25rem !important;
            font-weight: 700 !important;
        }}
        
        h2 {{
            font-size: 1.5rem !important;
        }}
        
        h3 {{
            font-size: 1.25rem !important;
        }}
        
        /* Paragraph text */
        p, span, label, .stMarkdown {{
            color: {text_primary} !important;
        }}
        
        .subtitle {{
            color: {text_secondary} !important;
            font-size: 1.1rem;
            font-weight: 400;
        }}

        /* Primary Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {accent_color} 0%, {accent_hover} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1.5rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
        }}
        
        .stButton > button:active {{
            transform: translateY(0) !important;
        }}

        /* Download button specific styling */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, {accent_color} 0%, {accent_hover} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
        }}

        /* Metric Cards */
        div[data-testid="metric-container"] {{
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            padding: 1.25rem !important;
            border-radius: 12px !important;
            box-shadow: {shadow} !important;
            transition: all 0.2s ease !important;
        }}
        
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: {shadow_hover} !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {text_secondary} !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {text_primary} !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }}

        /* Container/Card styling */
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 16px !important;
            box-shadow: {shadow} !important;
            transition: all 0.2s ease !important;
        }}
        
        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: {shadow_hover} !important;
        }}

        /* Images */
        img {{
            border-radius: 12px !important;
            transition: transform 0.3s ease !important;
        }}
        
        img:hover {{
            transform: scale(1.01) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {bg_secondary} !important;
            border-radius: 12px !important;
            padding: 4px !important;
            gap: 4px !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent !important;
            color: {text_secondary} !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.75rem 1.25rem !important;
            transition: all 0.2s ease !important;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            color: {text_primary} !important;
            background-color: {bg_card} !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {bg_card} !important;
            color: {accent_color} !important;
            box-shadow: {shadow} !important;
        }}
        
        /* Tab highlight bar */
        .stTabs [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}
        
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            background-color: {bg_secondary} !important;
            border: 2px dashed {border_color} !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            transition: all 0.2s ease !important;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {accent_color} !important;
            background-color: {info_bg} !important;
        }}
        
        [data-testid="stFileUploaderDropzone"] {{
            background-color: transparent !important;
        }}
        [data-testid="stFileUploaderDropzone"] div {{
            color: {text_primary} !important;
        }}
        
        [data-testid="stFileUploader"] button {{
            background: linear-gradient(135deg, {accent_color} 0%, {accent_hover} 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }}
        
        [data-testid="stFileUploader"] button * {{
            color: #ffffff !important;
        }}
        
        [data-testid="stFileUploader"] section {{
            padding: 0 !important;
        }}
        
        [data-testid="stFileUploader"] small {{
            color: {text_secondary} !important;
        }}

        /* DataFrames */
        [data-testid="stDataFrame"] {{
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid {border_color} !important;
        }}
        
        [data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {{
            background-color: {bg_card} !important;
        }}
        
        /* Custom Tables */
        .custom-table-wrapper {{
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid {border_color};
            background-color: {bg_card};
        }}
        
        .custom-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin: 0;
        }}
        
        .custom-table th {{
            background-color: {bg_secondary};
            color: {text_secondary};
            font-weight: 600;
            text-align: left;
            padding: 1rem;
            border-bottom: 2px solid {border_color};
            white-space: nowrap;
        }}
        
        .custom-table td {{
            padding: 1rem;
            color: {text_primary};
            border-bottom: 1px solid {border_color};
            background-color: {bg_card};
        }}
        
        .custom-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .custom-table tr:hover td {{
            background-color: {bg_secondary} !important;
        }}

        /* Alert boxes */
        .stAlert {{
            border-radius: 12px !important;
            border-width: 1px !important;
            font-weight: 500 !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="success"] {{
            background-color: {success_bg} !important;
            border-color: {success_border} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="success"] p {{
            color: {success_text} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {{
            background-color: {error_bg} !important;
            border-color: {error_border} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="error"] p {{
            color: {error_text} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {{
            background-color: {info_bg} !important;
            border-color: {info_border} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="info"] p {{
            color: {info_text} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {{
            background-color: {warning_bg} !important;
            border-color: {warning_border} !important;
        }}
        
        [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] p {{
            color: {warning_text} !important;
        }}

        /* Spinner */
        .stSpinner > div {{
            border-top-color: {accent_color} !important;
        }}

        /* Divider */
        hr {{
            border-color: {border_color} !important;
            margin: 1.5rem 0 !important;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {bg_secondary};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {text_secondary};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {text_primary};
        }}

        /* Theme toggle button */
        .theme-toggle {{
            position: fixed;
            top: 0.75rem;
            right: 1rem;
            z-index: 999999;
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            padding: 0.5rem 1rem !important;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: {shadow};
        }}
        
        .theme-toggle:hover {{
            box-shadow: {shadow_hover};
            transform: translateY(-1px);
        }}

        /* Header area styling */
        .header-container {{
            text-align: center;
            padding: 2rem 0 1rem 0;
        }}
        
        .header-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}
        
        .header-subtitle {{
            font-size: 1.1rem;
            color: {text_secondary};
            font-weight: 400;
        }}
        
        .header-icon {{
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }}
        
        .header-icon svg {{
            width: 80px;
            height: 80px;
        }}

        /* Badge styling */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .badge-success {{
            background-color: {success_bg};
            color: {success_text};
            border: 1px solid {success_border};
        }}
        
        .badge-error {{
            background-color: {error_bg};
            color: {error_text};
            border: 1px solid {error_border};
        }}

        /* Feature card in About section */
        .feature-card {{
            background-color: {bg_secondary};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }}
        
        .feature-card:hover {{
            transform: translateY(-2px);
            box-shadow: {shadow_hover};
        }}
        
        .feature-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 0.5rem;
        }}
        
        .feature-desc {{
            font-size: 0.875rem;
            color: {text_secondary};
            line-height: 1.5;
        }}

        /* Footer disclaimer */
        .footer-disclaimer {{
            text-align: center;
            color: {text_secondary};
            font-size: 0.8rem;
            padding: 1rem 0;
            border-top: 1px solid {border_color};
            margin-top: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -------- CONSTANTS & SETUP --------
HISTORY_FILE = "history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# -------- THEME TOGGLE BUTTON --------
theme_col1, theme_col2 = st.columns([6, 1])
with theme_col2:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    theme_label = "Dark" if st.session_state.theme == "light" else "Light"
    if st.button(f"{theme_icon} {theme_label}", key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()

# -------- APP HEADER --------
try:
    with open("Assets/icon.svg", "r", encoding="utf-8") as f:
        svg_icon = f.read()
except:
    svg_icon = "🩺"

st.markdown(f"""
    <div class="header-container">
        <div class="header-icon">{svg_icon}</div>
        <h1 class="header-title">CliniScan AI</h1>
        <p class="header-subtitle">Advanced Chest X-Ray Analysis & Diagnostics Platform</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------- NAVIGATION TABS --------
tab_dashboard, tab_history, tab_about = st.tabs(["Dashboard", "Analysis History", "About"])

# -------- DASHBOARD PAGE --------
with tab_dashboard:
    st.markdown("""
        <p style='text-align: center; opacity: 0.8; margin-bottom: 1.5rem;'>
            Upload a PA/AP view chest radiograph for deep learning-based automated screening.
        </p>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        upload_col, info_col = st.columns([2, 1])
        with upload_col:
            uploaded_file = st.file_uploader(
                "Drag and drop your X-ray image here",
                type=["png", "jpg", "jpeg"],
                help="Supported formats: PNG, JPG, JPEG"
            )
        with info_col:
            st.info("**Supported Formats**\n\nPNG, JPG, JPEG\n\nMax size: 200MB")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("### Diagnostic Results")
        
        with st.spinner("Analyzing scan with deep learning models..."):
            result = run_pipeline(image)
            
        col_img, col_res = st.columns([1, 1.2])
        
        with col_img:
            with st.container(border=True):
                st.image(image, caption="Original Input Radiograph", use_container_width=True)
            
        with col_res:
            with st.container(border=True):
                label = result['label']
                confidence = result['confidence']
                is_abnormal = (label == "Abnormal")
                
                m1, m2 = st.columns(2)
                m1.metric(
                    label="Primary Finding", 
                    value=label,
                    delta="Requires Attention" if is_abnormal else "Clear",
                    delta_color="inverse" if is_abnormal else "normal"
                )
                m2.metric(
                    label="AI Confidence", 
                    value=f"{confidence * 100:.1f}%",
                    delta="High" if confidence > 0.85 else "Review",
                    delta_color="normal"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if is_abnormal:
                    st.error("**Clinical Alert:** Abnormalities detected. Please review the visualizations below and consult a radiologist.")
                else:
                    st.success("**Screening Clear:** No significant pulmonary abnormalities detected.")

            with st.container(border=True):
                st.markdown("#### Download Report")
                path = generate_report(result)
                
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                    
                clean_name = os.path.splitext(uploaded_file.name)[0]
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"CliniScan_Report_{clean_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Advanced Visualizations")
        
        viz_tab1, viz_tab2 = st.tabs(["Localization (YOLOv8)", "Attention Heatmaps (Grad-CAM)"])
        
        with viz_tab1:
            with st.container(border=True):
                if is_abnormal:
                    if result.get("box_count", 0) > 0:
                        st.success(f"Localization successful: {result['box_count']} region(s) identified.")
                        st.info("Bounding boxes indicate regions of potential abnormalities.")
                        tcol1, tcol2, tcol3 = st.columns([1, 2, 1])
                        with tcol2:
                            st.image(result["image"], caption="Detected Regions", use_container_width=True)
                    else:
                        st.warning("Classification flagged as Abnormal, but localization model could not identify precise regions at current confidence threshold.")
                        tcol1, tcol2, tcol3 = st.columns([1, 2, 1])
                        with tcol2:
                            st.image(image, caption="Original Image", use_container_width=True)
                else:
                    st.info("No localized abnormalities detected by the object detection model.")
                
        with viz_tab2:
            with st.container(border=True):
                st.info("Grad-CAM highlights areas most critical to the classifier's decision. Warmer colors indicate higher importance.")
                
                with st.spinner("Generating attention heatmap..."):
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    heatmap = generate_gradcam(classifier, img_tensor)
                    overlay = overlay_gradcam(image, heatmap)
                    
                h1, h2, h3 = st.columns([1, 0.15, 1])
                with h1:
                    st.image(image, caption="Original X-ray", use_container_width=True)
                with h2:
                    st.markdown("<p style='text-align: center; margin-top: 50%; font-size: 1.5rem; opacity: 0.5;'>→</p>", unsafe_allow_html=True)
                with h3:
                    st.image(overlay, caption="Grad-CAM Activation", use_container_width=True)
            
        # -------- SAVE HISTORY --------
        with st.spinner("Saving to history..."):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            
            if not history or history[-1].get("filename") != uploaded_file.name:
                history.append({
                    "filename": uploaded_file.name,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                with open(HISTORY_FILE, "w") as f:
                    json.dump(history, f)


# -------- HISTORY PAGE --------
with tab_history:
    st.markdown("### Analysis History")
    st.markdown("View all previous scans and their results.")
    
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        
    if len(history) == 0:
        st.info("No analyses logged yet. Upload an X-ray on the Dashboard to get started.")
    else:
        df = pd.DataFrame(history)
        
        if "filename" not in df.columns:
            df["filename"] = "Unknown"
        if "date" not in df.columns:
            df["date"] = "Unknown"
            
        df = df[["date", "filename", "label", "confidence"]]
        df.rename(columns={
            "date": "Timestamp", 
            "filename": "Image Reference", 
            "label": "Diagnosis", 
            "confidence": "Confidence"
        }, inplace=True)
        
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
        
        # Statistics
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            total = len(df)
            abnormal = len(df[df["Diagnosis"] == "Abnormal"])
            normal = len(df[df["Diagnosis"] == "Normal"])
            
            col1.metric("Total Scans", total)
            col2.metric("Abnormal", abnormal)
            col3.metric("Normal", normal)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            html_table = df.iloc[::-1].to_html(index=False, classes="custom-table", escape=False, border=0)
            st.markdown(f'<div class="custom-table-wrapper">{html_table}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Clear History", type="secondary"):
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)
            st.rerun()

# -------- ABOUT PAGE --------
with tab_about:
    st.markdown("### About CliniScan AI")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("""
            #### Project Vision
            
            **CliniScan AI** is an advanced deep learning-powered clinical assistant designed to analyze 
            Chest Radiographs (X-Rays). It provides medical professionals with rapid screening capabilities 
            prioritizing speed, transparency, and clinical usability.
            
            ---
            
            #### Core Capabilities
            
            **Lesion Localization**  
            YOLOv8-powered spatial detection of infiltrates and anomalies with high precision.
            
            **Binary Classification**  
            Normal vs. Abnormal differentiation using custom CNN architectures built with PyTorch.
            
            **Explainable AI**  
            Grad-CAM visualizations showing which areas influenced the model's diagnosis.
            
            **Automated Reporting**  
            End-to-end generation of standardized PDF reports for patient records.
            """)
            
    with col2:
        with st.container(border=True):
            st.markdown("#### Academic Context")
            st.markdown("""
            Developed as part of a Master's Thesis in Artificial Intelligence. 
            This multi-model pipeline serves as an exploratory framework for medical computer vision.
            """)
            
        with st.container(border=True):
            st.markdown("#### Technology Stack")
            st.markdown("""
            - **Deep Learning:** PyTorch, Torchvision
            - **Computer Vision:** Ultralytics, OpenCV, PIL
            - **Interface:** Streamlit
            - **Data:** Pandas, JSON
            """)
    
    st.markdown("""
        <div class="footer-disclaimer">
            <strong>Disclaimer:</strong> Unauthorized clinical use is prohibited. 
            Intended solely for academic evaluation and research purposes.
        </div>
    """, unsafe_allow_html=True)
