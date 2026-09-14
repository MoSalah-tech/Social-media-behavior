from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import predict_one, load_model

app = FastAPI(
    title="Influencer Detection API",
    description="Predict whether a social media user is an influencer from behavior features.",
    version="1.0.0",
)


class UserFeatures(BaseModel):
    model_config = {"extra": "allow"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    _, _, schema = load_model()
    return {
        "target": schema["target"],
        "task": schema["task"],
        "classes": schema["classes"],
        "metrics": schema["metrics"],
    }


@app.post("/predict")
def predict(payload: UserFeatures):
    try:
        row = payload.model_dump(exclude_none=True)
        return predict_one(row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
