import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {'message': 'Predicting valence from brain signals'}


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
