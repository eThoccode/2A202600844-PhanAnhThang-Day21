from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

DEFAULT_S3_MODEL_KEY = "models/latest/model.pkl"
DEFAULT_MODEL_PATH = os.path.expanduser("~/models/model.pkl")
FEATURE_COUNT = 12
LABELS = {
    0: "thap",
    1: "trung_binh",
    2: "cao",
}
model = None


def get_settings():
    cloud_bucket = os.environ.get("CLOUD_BUCKET")
    if not cloud_bucket:
        raise RuntimeError("CLOUD_BUCKET environment variable is required")
    return {
        "cloud_bucket": cloud_bucket,
        "s3_model_key": os.environ.get("S3_MODEL_KEY", DEFAULT_S3_MODEL_KEY),
        "model_path": os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
    }


def download_model():
    settings = get_settings()
    os.makedirs(os.path.dirname(settings["model_path"]), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(settings["cloud_bucket"], settings["s3_model_key"], settings["model_path"])
    print(
        f"Model da duoc tai xuong tu s3://{settings['cloud_bucket']}/{settings['s3_model_key']}."
    )
    return settings["model_path"]


def load_model():
    global model
    model_path = download_model()
    model = joblib.load(model_path)
    print(f"Model da duoc nap tu {model_path}.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if len(req.features) != FEATURE_COUNT:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])
    label = LABELS.get(prediction)
    if label is None:
        raise HTTPException(status_code=500, detail="Unexpected prediction label")

    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
