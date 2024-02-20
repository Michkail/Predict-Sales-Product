import uvicorn

from fastapi import FastAPI, HTTPException, Depends, status, Body
from pydantic import BaseModel
from typing import List
from starlette.responses import JSONResponse
from tests.model import train, predict

app = FastAPI()
trained_model_path = "./productsalespredict"


class TrainingData(BaseModel):
    save_path: str
    data: List[dict]


class PredictionData(BaseModel):
    category: str
    price: float
    promotion: int
    discount_perc: float
    channel: str


@app.get("/")
async def read_root():
    return {"status": "Service is up and running."}


@app.post("/train")
async def train_model(training_data: TrainingData):
    try:
        accuracy = train(training_data.data, training_data.save_path)
        return JSONResponse(content={"accuracy": accuracy, "status": "success"}, status_code=status.HTTP_200_OK)

    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)},
                            status_code=status.HTTP_412_PRECONDITION_FAILED)


@app.post("/predict")
async def make_prediction(prediction_data: PredictionData):
    try:
        result = predict(prediction_data.dict(), trained_model_path)

        if result == 'Success':
            return JSONResponse(content={"result": result, "status": "success"},
                                status_code=status.HTTP_200_OK)

        else:
            return JSONResponse(content={"result": result, "success": "success"},
                                status_code=status.HTTP_417_EXPECTATION_FAILED)

    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)},
                            status_code=status.HTTP_412_PRECONDITION_FAILED)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
