"""Configuration values for the face verification demo."""

MODEL_PACK_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.40

REFERENCE_LABEL = "Reference Photo"
VERIFY_LABEL = "Photo to Verify"

SAMPLE_PAIRS = {
    "Clear match pair": {
        "reference": "sample_data/match_pair/ref.jpg",
        "verify": "sample_data/match_pair/verify.jpg",
    },
    "Clear non-match pair": {
        "reference": "sample_data/no_match_pair/ref.jpg",
        "verify": "sample_data/no_match_pair/verify.jpg",
    },
}
