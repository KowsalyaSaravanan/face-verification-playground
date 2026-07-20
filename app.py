from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import REFERENCE_LABEL, SAMPLE_PAIRS, SIMILARITY_THRESHOLD, VERIFY_LABEL
from src.face_utils import (
    IdentityChecker,
    load_face_detector,
    load_image_path,
    read_image_file,
    validate_single_face,
)


ROOT_DIR = Path(__file__).parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --kyc-ink: #111827;
            --kyc-muted: #6b7280;
            --kyc-line: #e5e7eb;
            --kyc-soft: #f7f8fb;
            --kyc-green: #047857;
            --kyc-red: #dc2626;
            --kyc-blue: #1d4ed8;
            --kyc-teal: #0f766e;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.08), transparent 34%),
                linear-gradient(180deg, #eef2f7 0%, #ffffff 38%);
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.86);
            backdrop-filter: blur(10px);
        }
        h1, h2, h3, p {
            letter-spacing: 0;
        }
        div.stButton > button[kind="primary"] {
            width: 260px;
            height: 48px;
            border-radius: 6px;
            background: #111827;
            border: 1px solid #111827;
            font-weight: 700;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1f2937;
            border-color: #1f2937;
        }
        .hero {
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 0;
            background: #0f172a;
            box-shadow: 0 22px 48px rgba(15, 23, 42, 0.18);
            margin-bottom: 18px;
            overflow: hidden;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.7fr) minmax(300px, 0.8fr);
            gap: 28px;
            padding: 34px 36px;
        }
        .hero::before {
            content: "";
            display: block;
            height: 5px;
            background: linear-gradient(90deg, #14b8a6, #38bdf8, #a7f3d0);
        }
        .eyebrow {
            color: #5eead4;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(2.15rem, 4vw, 3.55rem);
            line-height: 1.05;
            margin: 0;
        }
        .hero p {
            color: #cbd5e1;
            max-width: 720px;
            font-size: 1rem;
            margin: 14px 0 0 0;
        }
        .hero-card {
            border: 1px solid rgba(148, 163, 184, 0.36);
            background: rgba(15, 23, 42, 0.68);
            border-radius: 8px;
            padding: 18px;
        }
        .hero-card h3 {
            color: #ffffff;
            margin: 0 0 12px 0;
            font-size: 1rem;
        }
        .check-row {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            color: #cbd5e1;
            border-top: 1px solid rgba(148, 163, 184, 0.22);
            padding: 10px 0;
            font-size: 0.88rem;
        }
        .check-row:first-of-type {
            border-top: 0;
        }
        .check-row strong {
            color: #ffffff;
        }
        .privacy {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
            border-radius: 6px;
            padding: 12px 14px;
            margin: 8px 0 18px 0;
            font-size: 0.92rem;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 14px 0 18px 0;
        }
        .workflow-card {
            border: 1px solid var(--kyc-line);
            background: #ffffff;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .workflow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            background: #ccfbf1;
            color: #115e59;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .workflow-card strong {
            display: block;
            color: var(--kyc-ink);
            margin-bottom: 4px;
        }
        .workflow-card p {
            color: var(--kyc-muted);
            margin: 0;
            font-size: 0.9rem;
        }
        .panel-heading {
            font-size: 0.84rem;
            font-weight: 800;
            color: var(--kyc-teal);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }
        .section-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--kyc-ink);
            margin: 0 0 4px 0;
        }
        .section-subtitle {
            font-size: 0.88rem;
            color: var(--kyc-muted);
            margin: 0 0 14px 0;
        }
        .status-ok {
            border: 1px solid #bbf7d0;
            background: #f0fdf4;
            color: #166534;
            border-radius: 6px;
            padding: 10px 12px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .status-wait {
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            color: #4b5563;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        .badge {
            display: inline-block;
            padding: 0.52rem 0.8rem;
            border-radius: 6px;
            color: white;
            font-weight: 800;
            margin-top: 0.25rem;
        }
        .badge.match { background: var(--kyc-green); }
        .badge.no-match { background: var(--kyc-red); }
        .result-panel {
            border: 1px solid var(--kyc-line);
            background: #ffffff;
            border-radius: 8px;
            padding: 18px;
            margin-top: 14px;
        }
        @media (max-width: 900px) {
            .hero-grid,
            .workflow-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading InsightFace models. First run can take a few minutes...")
def get_face_detector():
    return load_face_detector()


def show_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Identity verification console</div>
                    <h1>Face Verification Playground</h1>
                    <p>A public portfolio-grade recreation of a KYC face-match screen. The app
                    accepts a reference photo and a live or uploaded verification photo, validates
                    that each contains exactly one face, then compares InsightFace embeddings.</p>
                </div>
                <div class="hero-card">
                    <h3>Session controls</h3>
                    <div class="check-row"><span>Storage</span><strong>In memory only</strong></div>
                    <div class="check-row"><span>Detector</span><strong>InsightFace CPU</strong></div>
                    <div class="check-row"><span>Decision</span><strong>Cosine threshold</strong></div>
                    <div class="check-row"><span>Use case</span><strong>Portfolio demo</strong></div>
                </div>
            </div>
        </div>
        <div class="privacy">Photos are processed only in memory for this session. Nothing is saved,
        stored in a database, or kept after the session ends.</div>
        """,
        unsafe_allow_html=True,
    )


def show_workflow() -> None:
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-card">
                <div class="workflow-number">1</div>
                <strong>Capture both photos</strong>
                <p>Use file upload or camera capture independently for reference and verify images.</p>
            </div>
            <div class="workflow-card">
                <div class="workflow-number">2</div>
                <strong>Validate one face</strong>
                <p>Each photo must contain exactly one detected face before verification can run.</p>
            </div>
            <div class="workflow-card">
                <div class="workflow-number">3</div>
                <strong>Compare identity</strong>
                <p>InsightFace embeddings are compared with normalized cosine similarity.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_input(label: str, key_prefix: str):
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload an image or capture a live webcam photo.</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Input method",
        ["Upload", "Use Camera"],
        key=f"{key_prefix}_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Upload":
        uploaded = st.file_uploader(
            f"{label} upload",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload",
            label_visibility="collapsed",
        )
        if uploaded is None:
            return None
        return read_image_file(uploaded)

    captured = st.camera_input(
        f"{label} camera capture",
        key=f"{key_prefix}_camera",
        label_visibility="collapsed",
    )
    if captured is None:
        return None
    return read_image_file(captured)


def validate_and_render(face_detector, label: str, image_rgb):
    if image_rgb is None:
        st.markdown(
            f'<div class="status-wait">Waiting for {label.lower()}.</div>',
            unsafe_allow_html=True,
        )
        return None

    try:
        validation = validate_single_face(face_detector, image_rgb)
    except Exception as exc:
        st.error(f"Could not process this photo: {exc}")
        return None

    if validation.error:
        st.error(validation.error)
        st.image(image_rgb, caption="Retry with upload or camera.", width="stretch")
        return validation

    st.markdown('<div class="status-ok">Exactly one face detected.</div>', unsafe_allow_html=True)
    st.image(validation.preview_rgb, caption="Detected face region", width="stretch")
    return validation


def load_demo_pair(pair_name: str):
    pair = SAMPLE_PAIRS[pair_name]
    reference = load_image_path(str(ROOT_DIR / pair["reference"]))
    verify = load_image_path(str(ROOT_DIR / pair["verify"]))
    return reference, verify


def result_badge(is_match: bool) -> None:
    label = "Match" if is_match else "No Match"
    css_class = "match" if is_match else "no-match"
    st.markdown(
        f'<div class="badge {css_class}">{label}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Face Verification Playground",
        page_icon="ID",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    show_header()
    show_workflow()

    face_detector = get_face_detector()

    controls = st.container(border=True)
    with controls:
        st.markdown('<div class="panel-heading">Verification setup</div>', unsafe_allow_html=True)
        control_col_1, control_col_2 = st.columns([1, 2])
        with control_col_1:
            demo_mode = st.checkbox("Load sample photos", value=False)
        with control_col_2:
            threshold = st.slider(
                "Match threshold",
                min_value=0.20,
                max_value=0.70,
                value=float(SIMILARITY_THRESHOLD),
                step=0.01,
            )

    if demo_mode:
        pair_name = st.selectbox("Sample pair", list(SAMPLE_PAIRS.keys()))
        reference_image, verify_image = load_demo_pair(pair_name)
    else:
        reference_image = verify_image = None

    ref_col, verify_col = st.columns(2)
    with ref_col:
        with st.container(border=True):
            if not demo_mode:
                reference_image = image_input(REFERENCE_LABEL, "reference")
            else:
                st.markdown(f'<div class="section-title">{REFERENCE_LABEL}</div>', unsafe_allow_html=True)
            reference_validation = validate_and_render(face_detector, REFERENCE_LABEL, reference_image)

    with verify_col:
        with st.container(border=True):
            if not demo_mode:
                verify_image = image_input(VERIFY_LABEL, "verify")
            else:
                st.markdown(f'<div class="section-title">{VERIFY_LABEL}</div>', unsafe_allow_html=True)
            verify_validation = validate_and_render(face_detector, VERIFY_LABEL, verify_image)

    can_verify = bool(
        reference_validation
        and verify_validation
        and reference_validation.is_valid
        and verify_validation.is_valid
    )

    verify_clicked = st.button("Verify", type="primary", disabled=not can_verify)
    if not can_verify:
        st.caption("Verify becomes active after both photos have exactly one detected face.")

    if verify_clicked and can_verify:
        try:
            identity_checker = IdentityChecker(face_detector, threshold=threshold)
            reference_embedding = reference_validation.embedding
            verify_embedding = verify_validation.embedding
            is_match, score = identity_checker.verify_embeddings(reference_embedding, verify_embedding)
        except Exception as exc:
            st.error(f"Verification failed: {exc}")
            return

        similarity_percent = max(0.0, min(1.0, score)) * 100

        with st.container(border=True):
            st.markdown('<div class="panel-heading">Verification result</div>', unsafe_allow_html=True)
            result_col_1, result_col_2 = st.columns([1, 2])
            with result_col_1:
                st.metric("Similarity score", f"{similarity_percent:.2f}%")
                result_badge(is_match)
                st.caption(f"Decision threshold: {threshold:.2f} cosine similarity")
            with result_col_2:
                photo_col_1, photo_col_2 = st.columns(2)
                with photo_col_1:
                    st.image(reference_validation.image_rgb, caption="Original reference photo", width="stretch")
                with photo_col_2:
                    st.image(verify_validation.image_rgb, caption="Original photo to verify", width="stretch")

        with st.expander("See what the model actually compared"):
            crop_col_1, crop_col_2 = st.columns(2)
            with crop_col_1:
                st.image(reference_validation.crop_rgb, caption="Reference face-only crop", width="stretch")
            with crop_col_2:
                st.image(verify_validation.crop_rgb, caption="Verify face-only crop", width="stretch")


if __name__ == "__main__":
    main()
