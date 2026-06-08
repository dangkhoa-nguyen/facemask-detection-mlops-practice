from fastapi import FastAPI, UploadFile, File
from pathlib import Path

from src.inference.predictor import predict_image

app = FastAPI(
    title="Face Mask Detection API",
    version="1.0.0"
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Face Mask Detection API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_path = UPLOAD_DIR / file.filename

    with open(image_path, "wb") as f:
        f.write(await file.read())

    result = predict_image(str(image_path))

    return result["result_json"]