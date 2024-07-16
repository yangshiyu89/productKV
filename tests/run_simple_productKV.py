from PIL import Image
from productKV import ProductKV
    
    
sd15_base_path = "/workspace/models/sd15/realistic-vision-v51"
rmbg_path = "/workspace/models/RMBG-1.4"
ic_light_path = "/workspace/models/ic_light/iclight_sd15_fc.safetensors"


image_path = "images/sk2.jpg"
image = Image.open(image_path).convert("RGB")
product_prompt = "A set of cosmetics"
background_prompt = "city background"
image_width = 1024
image_height = 1024
num_samples = 1
seed = 12345
steps = 25
a_prompt = "best quality"
n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
cfg = 2


product_obj = ProductKV(device="cuda:0")
product_obj.load_models(
    sd15_base_path,
    rmbg_path,
    ic_light_path 
)

result = product_obj(
    image, 
    product_prompt, 
    background_prompt, 
    image_width, 
    image_height, 
    num_samples, 
    seed, 
    steps, 
    a_prompt, 
    n_prompt, 
    cfg
)

result.save("result.png")