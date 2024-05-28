import onnxruntime.backend
# from definitions import ROOT_DIR
from dotenv import dotenv_values


name = "Face Detection"
docs_url = "/docs"
version = "1.0.0"

ACCESSORY = "accessory"
FACE_ID_CARD_QUALITY = "face_id_card_quality"
FACE_DETECT = "SCRFD"


PATH_MODEL = {
    FACE_DETECT: "weight/det_10g.onnx",
}

# dotenv_values = dotenv_values(f"{ROOT_DIR}/.env")
# provider = dotenv_values.get("PROVIDERS", "gpu")

# if provider == "gpu":
#     PROVIDERS = ["CUDAExecutionProvider"]
# else:
PROVIDERS = ["CPUExecutionProvider"]
MODEL = {
    # ACCESSORY: onnxruntime.InferenceSession(
    #     PATH_MODEL[ACCESSORY], providers=PROVIDERS
    # ),
    # FACE_ID_CARD_QUALITY: onnxruntime.InferenceSession(
    #     PATH_MODEL[FACE_ID_CARD_QUALITY], providers=PROVIDERS
    # ),
    FACE_DETECT: onnxruntime.InferenceSession(
        PATH_MODEL[FACE_DETECT], providers=PROVIDERS
    ),
}
