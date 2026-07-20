# Face Verification Playground

A lightweight KYC-style face verification demo built with Streamlit and InsightFace.

## Live Demo

[Open the live Streamlit demo](https://face-verification-playground-y6njm5svnwtmynouybvhys.streamlit.app/)

## What It Does

This app compares two photos and decides whether they appear to show the same person. It recreates the face-matching core of a KYC identity-check flow using only open InsightFace models, synthetic sample images, and session-only processing.

Users can provide each photo by upload or webcam capture, so all combinations are supported:

1. Upload vs upload
2. Upload vs webcam
3. Webcam vs upload
4. Webcam vs webcam

## How It Works

1. Detect one face in each full photo with InsightFace.
2. Crop only the detected face region from each photo.
3. Generate embeddings from those face-only crops.
4. Compare the embeddings with normalized cosine similarity and apply the same default threshold style as the backend `IdentityChecker`.

The app rejects images with zero faces or multiple faces and asks the user to retry that specific photo.

The app logic is intentionally close to the existing backend pattern:

- `FaceDetector` wraps `insightface.app.FaceAnalysis()`.
- The model runs on CPU with `ctx_id=-1` and `det_size=(640, 640)`.
- `IdentityChecker` normalizes both embeddings before the dot-product similarity check.
- Default match threshold is `0.60`, adjustable from the sidebar.

## Screenshots/GIF

Add screenshots after running or deploying the app:

- Upload-vs-upload demo using the bundled sample pair.
- Webcam capture demo using `st.camera_input`.

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

The first run may take a few minutes because InsightFace downloads the open `buffalo_l` model pack automatically.

This project includes `runtime.txt` with `python-3.12` to match the local backend/runtime target.

On Windows with Python 3.12, `insightface==0.7.3` may need Microsoft Visual C++ Build Tools because it builds a native extension when no matching wheel is available.

## Demo Mode

The sidebar includes a bundled sample-photo mode with:

- A clear match pair using synthetic images.
- A clear non-match pair using synthetic images.

No real customer, bank, or proprietary images are included.

## Privacy

Photos are processed only in memory for the current Streamlit session. The app does not use a database, does not create accounts, and does not persist uploaded or captured images after the session ends.

## Deployment Steps

Streamlit Community Cloud deployment must be completed by the author in their browser session:

1. Push this repo to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Choose **Deploy a public app from GitHub**.
4. Select repository, branch `main`, and main file path `app.py`.
5. Leave secrets empty because this app needs no API keys.
6. Wait for the first model download and app startup to finish.
7. Copy the verified `*.streamlit.app` URL into the Live Demo section above.

`requirements.txt` uses `opencv-python-headless`, which is required for Streamlit Cloud compatibility.
`runtime.txt` pins Python 3.12 to match the target runtime.

## Limitations

This is an educational portfolio demo, not a production identity-verification system. It does not include liveness checks, anti-spoofing, fraud workflows, manual review queues, audit logging, or compliance controls.

## Tech Stack

- Python
- Streamlit
- InsightFace `0.7.3`
- ONNX Runtime `1.20.1`
- OpenCV headless `4.10.0.84`
- Pillow `11.0.0`
- NumPy `1.26.4`

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
├── README.md
├── LICENSE
├── .gitignore
└── .env.example
```

## License

MIT License. See [LICENSE](LICENSE).
