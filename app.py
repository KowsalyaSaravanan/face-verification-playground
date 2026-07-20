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
        }
        .stApp {
            background: linear-gradient(180deg, #f6f8fb 0%, #ffffff 32%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
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
            width: 220px;
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
            border: 1px solid var(--kyc-line);
            border-radius: 8px;
            padding: 28px 30px;
            background: #ffffff;
            box-shadow: 0 18px 40px rgba(17, 24, 39, 0.07);
            margin-bottom: 20px;
        }
        .eyebrow {
            color: var(--kyc-blue);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero h1 {
            color: var(--kyc-ink);
            font-size: clamp(2rem, 4vw, 3.3rem);
            line-height: 1.05;
            margin: 0;
        }
        .hero p {
            color: var(--kyc-muted);
            max-width: 760px;
            font-size: 1rem;
            margin: 14px 0 0 0;
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
            <div class="eyebrow">KYC face match demo</div>
            <h1>Face Verification Playground</h1>
            <p>Compare a reference photo with a verification photo using InsightFace detection,
            face-only review crops, and normalized embedding similarity.</p>
        </div>
        <div class="privacy">Photos are processed only in memory for this session. Nothing is saved,
        stored in a database, or kept after the session ends.</div>
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

    face_detector = get_face_detector()

    controls = st.container(border=True)
    with controls:
        control_col_1, control_col_2 = st.columns([1, 2])
        with control_col_1:
            demo_mode = st.toggle("Use bundled sample photos", value=False)
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

        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.subheader("Verification Result")
        photo_col_1, photo_col_2 = st.columns(2)
        with photo_col_1:
            st.image(reference_validation.image_rgb, caption="Original reference photo", width="stretch")
        with photo_col_2:
            st.image(verify_validation.image_rgb, caption="Original photo to verify", width="stretch")

        st.metric("Similarity score", f"{similarity_percent:.2f}%")
        result_badge(is_match)
        st.caption(f"Decision threshold: {threshold:.2f} cosine similarity")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("See what the model actually compared"):
            crop_col_1, crop_col_2 = st.columns(2)
            with crop_col_1:
                st.image(reference_validation.crop_rgb, caption="Reference face-only crop", width="stretch")
            with crop_col_2:
                st.image(verify_validation.crop_rgb, caption="Verify face-only crop", width="stretch")


if __name__ == "__main__":
    main()
