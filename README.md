<div align="center">
  <img src="docs/assets/app-preview.png" alt="Face Verification Playground preview" width="100%" />
  <h1>Face Verification Playground</h1>
  <p>
    A portfolio-grade KYC-style face verification application built with Streamlit and InsightFace.
  </p>
  <p>
    <a href="https://face-verification-playground-y6njm5svnwtmynouybvhys.streamlit.app/"><strong>Live Streamlit Demo</strong></a>
    &nbsp;|&nbsp;
    <a href="#setup"><strong>Local Setup</strong></a>
    &nbsp;|&nbsp;
    <a href="#privacy"><strong>Privacy Notes</strong></a>
  </p>
</div>

---

## Application Overview

Face Verification Playground recreates the core face-matching workflow used in a KYC identity check:

<table>
  <tr>
    <td width="33%"><strong>1. Capture</strong><br />Accept a reference photo and a verification photo from upload or live webcam capture.</td>
    <td width="33%"><strong>2. Validate</strong><br />Run InsightFace detection and require exactly one face in each image before verification.</td>
    <td width="33%"><strong>3. Compare</strong><br />Generate embeddings and compare them with normalized cosine similarity.</td>
  </tr>
</table>

The project is a public, safe portfolio recreation. It does not include client code, bank data, proprietary weights, authentication, or persistent storage.

## Reference Images

The repository includes synthetic sample images so recruiters can try the app immediately without providing their own photos.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/assets/reference-sample.jpg" alt="Reference sample" width="100%" />
      <br />
      <strong>Reference Photo</strong>
    </td>
    <td align="center" width="50%">
      <img src="docs/assets/verify-sample.jpg" alt="Verification sample" width="100%" />
      <br />
      <strong>Photo to Verify</strong>
    </td>
  </tr>
</table>

## Product Behavior

- Each side supports both upload and live camera capture.
- A sample-photo mode is included for instant recruiter demos.
- The reference and verification images are validated independently.
- Zero-face images show a clear retry message.
- Multi-face images are rejected instead of auto-selecting a face.
- The Verify button stays disabled until both photos contain exactly one detected face.
- The result screen shows original photos, similarity score, Match or No Match verdict, and the face regions used for review.

## Implementation Notes

The face logic follows the same simple backend pattern used in production-style services:

- `FaceDetector` wraps `insightface.app.FaceAnalysis()`.
- The detector runs on CPU with `ctx_id=-1` and `det_size=(640, 640)`.
- `IdentityChecker` normalizes embeddings before dot-product cosine comparison.
- Default match threshold is `0.60`, adjustable in the app.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Using `uv` with Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv run streamlit run app.py
```

The first run can take a few minutes because InsightFace downloads the open model pack automatically.

On Windows with Python 3.12, `insightface==0.7.3` may require Microsoft Visual C++ Build Tools when a matching wheel is unavailable.

## Tech Stack

<table>
  <tr><td><strong>Frontend</strong></td><td>Streamlit</td></tr>
  <tr><td><strong>Face analysis</strong></td><td>InsightFace 0.7.3</td></tr>
  <tr><td><strong>Runtime</strong></td><td>Python 3.12</td></tr>
  <tr><td><strong>Inference backend</strong></td><td>ONNX Runtime 1.20.1 CPU</td></tr>
  <tr><td><strong>Image handling</strong></td><td>OpenCV headless, Pillow, NumPy</td></tr>
</table>

## Privacy

Photos are processed only in memory for the active Streamlit session. The app does not use a database, user accounts, cloud storage, or persistent file storage for uploaded or captured images.

## Limitations

This is an educational and portfolio demonstration, not a production identity-verification system. It does not include liveness checks, anti-spoofing, fraud review workflows, audit logging, or compliance controls.

## Project Structure

```text
face-verification-playground/
├── app.py
├── requirements.txt
├── runtime.txt
├── src/
│   ├── face_utils.py
│   └── config.py
├── sample_data/
│   ├── match_pair/
│   │   ├── ref.jpg
│   │   └── verify.jpg
│   └── no_match_pair/
│       ├── ref.jpg
│       └── verify.jpg
├── docs/
│   └── assets/
│       ├── app-preview.png
│       ├── reference-sample.jpg
│       └── verify-sample.jpg
├── README.md
├── LICENSE
├── .gitignore
└── .env.example
```

## License

MIT License. See [LICENSE](LICENSE).
