from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from image_generator import ImageGenerator
from starlette.responses import JSONResponse

app = FastAPI()

class Params(BaseModel):
    prompt: str
    num_samples: int
    image_resolution: int
    strength: float
    guess_mode: bool
    low_threshold: int
    high_threshold: int
    ddim_steps: int
    scale: float
    seed: int
    eta: float
    a_prompt: str
    n_prompt: str

class ImageData(BaseModel):
    image_array: list

@app.post("/generate")
async def generate_images(params: Params, image_data: ImageData):
    try:
        np_array = np.array(image_data.image_array, dtype=np.uint8)
        input_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if input_image is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        image_generator = ImageGenerator(params, input_image)
        generated_images = image_generator.generate()
        
        encoded_images = []
        for img in generated_images:
            _, buffer = cv2.imencode(".png", img)
            encoded_images.append(base64.b64encode(buffer).decode("utf-8"))
        
        return JSONResponse(content={"images": encoded_images})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
