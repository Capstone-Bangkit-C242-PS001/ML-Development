from fastapi import FastAPI, Request
import os
import utils
from utils import MF
import tensorflow as tf
import uvicorn

app = FastAPI()


@app.get("/")
async def hello_world():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    predict = utils.recommender(body['user_id'], body['skillset'])
    return predict


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
