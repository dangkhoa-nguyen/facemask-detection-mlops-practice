from fastapi import FastAPI

app = FastAPI(
    title="Face Mask Detection API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Face Mask Detection API"
    }