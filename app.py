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


@st.cache_resource(show_spinner="Loading InsightFace models. First run can take a few minutes...")
def get_face_detector():
    return load_face_detector()


def show_privacy_note() -> None:
    st.sidebar.caption(
        "Photos are processed only in memory for this session and are not saved or stored anywhere."
    )


def image_input(label: str, key_prefix: str):
    st.subheader(label)
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
        st.info(f"Add a {label.lower()} to continue.")
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

    st.success("Exactly one face detected.")
    st.image(validation.preview_rgb, caption="Detected face region", width="stretch")
    return validation


def load_demo_pair(pair_name: str):
    pair = SAMPLE_PAIRS[pair_name]
    reference = load_image_path(str(ROOT_DIR / pair["reference"]))
    verify = load_image_path(str(ROOT_DIR / pair["verify"]))
    return reference, verify


def result_badge(is_match: bool) -> None:
    color = "#16803c" if is_match else "#b42318"
    label = "Match" if is_match else "No Match"
    st.markdown(
        f"""
        <div style="display:inline-block;padding:0.45rem 0.8rem;border-radius:6px;
        background:{color};color:white;font-weight:700;margin-top:0.25rem;">{label}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Face Verification Playground", page_icon="ID", layout="wide")
    st.title("Face Verification Playground")
    st.caption("KYC-style two-photo face verification with InsightFace.")
    show_privacy_note()

    face_detector = get_face_detector()

    st.sidebar.header("Demo")
    demo_mode = st.sidebar.toggle("Use bundled sample photos", value=True)
    threshold = st.sidebar.slider(
        "Match threshold",
        min_value=0.20,
        max_value=0.70,
        value=float(SIMILARITY_THRESHOLD),
        step=0.01,
    )

    if demo_mode:
        pair_name = st.sidebar.selectbox("Sample pair", list(SAMPLE_PAIRS.keys()))
        reference_image, verify_image = load_demo_pair(pair_name)
    else:
        reference_image = verify_image = None

    ref_col, verify_col = st.columns(2)
    with ref_col:
        if not demo_mode:
            reference_image = image_input(REFERENCE_LABEL, "reference")
        reference_validation = validate_and_render(face_detector, REFERENCE_LABEL, reference_image)

    with verify_col:
        if not demo_mode:
            verify_image = image_input(VERIFY_LABEL, "verify")
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
            reference_embedding = identity_checker.get_embedding(reference_validation.crop_rgb)
            verify_embedding = identity_checker.get_embedding(verify_validation.crop_rgb)
            is_match, score = identity_checker.verify_embeddings(reference_embedding, verify_embedding)
        except Exception as exc:
            st.error(f"Verification failed: {exc}")
            return

        similarity_percent = max(0.0, min(1.0, score)) * 100

        st.divider()
        st.header("Verification Result")
        photo_col_1, photo_col_2 = st.columns(2)
        with photo_col_1:
            st.image(reference_validation.image_rgb, caption="Original reference photo", width="stretch")
        with photo_col_2:
            st.image(verify_validation.image_rgb, caption="Original photo to verify", width="stretch")

        st.metric("Similarity score", f"{similarity_percent:.2f}%")
        result_badge(is_match)
        st.caption(f"Decision threshold: {threshold:.2f} cosine similarity")

        with st.expander("See what the model actually compared"):
            crop_col_1, crop_col_2 = st.columns(2)
            with crop_col_1:
                st.image(reference_validation.crop_rgb, caption="Reference face-only crop", width="stretch")
            with crop_col_2:
                st.image(verify_validation.crop_rgb, caption="Verify face-only crop", width="stretch")


if __name__ == "__main__":
    main()
