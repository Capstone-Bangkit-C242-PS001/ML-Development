from fastapi import FastAPI, Request
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
    uvicorn.run(app, host = 'localhost', port = 8000)