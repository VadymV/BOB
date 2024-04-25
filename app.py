import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image
from typing import Tuple

from starlette.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("main.html", "r") as file:
        return file.read()


def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((180, 180), resample=Image.BICUBIC)
    image = np.array(img_resized)
    return image, img_resized.size


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image, img_size = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        return {
            'class': 1,
            'confidence': 1
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("main.html", "r") as file:
        return file.read()


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000)

