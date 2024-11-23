from fastapi import FastAPI, Request
import utils

app = FastAPI()

@app.get("/")
async def hello_world():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    predict = utils.recommender(body['user_id'], body['skillset'])
    return predict