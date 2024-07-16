import argparse
import base64
from PIL import Image
from io import BytesIO
import os
import uvicorn
from fastapi import File, UploadFile, Request, FastAPI
from productKV import ProductKV

parser = argparse.ArgumentParser()
parser.add_argument('--port', default=8501, type=int)
args = parser.parse_args()
port = args.port


sd15_base_path = "/workspace/models/sd15/realistic-vision-v51"
rmbg_path = "/workspace/models/RMBG-1.4"
ic_light_path = "/workspace/models/ic_light/iclight_sd15_fc.safetensors"

def base642pil(base64_data):
    base64_data = base64_data.encode("utf-8")
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    return Image.open(image_data).convert("RGB")

def load_ckpt():
    product_obj = ProductKV()
    
    product_obj.load_models(
        sd15_base_path = sd15_base_path,
        rmbg_path = rmbg_path,
        ic_light_path = ic_light_path,
    )
    return product_obj

product_obj = load_ckpt()

app = FastAPI()
save_dirs = "./demos/output/"
os.makedirs(save_dirs, exist_ok=True)

@app.post('/productKV')
async def productKV(request: Request):
    params = await request.json()
    product_image_base64 = params.pop('product_images')
    product_image = base642pil(product_image_base64)
    
    product_prompt = params.pop("product_prompt")
    background_prompt = params.pop("background_prompt")
    height = params.pop("height")
    width = params.pop("width")
    save_name = params.pop("save_name")
    
    product_image = product_image.resize((width, height))
    
    result_image = product_obj(
        image = product_image,
        product_prompt = product_prompt, 
        background_prompt = background_prompt,
        image_height = height,
        image_width = width,
    )
    
    result_image.save(save_dirs + '{}.png'.format(save_name))
    return {
        "message": "OK"
    }
    
if __name__ == '__main__':
    uvicorn.run(app, port=port, host='0.0.0.0')