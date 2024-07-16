import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import time
import uuid
import requests
import os

def streamlit_image2base64(image_file):
    image_file = Image.open(image_file)
    output_buffer = BytesIO()
    image_file.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    image_file = base64.b64encode(byte_data)
    return image_file.decode("utf-8") 

st.set_page_config(layout="wide")

args = {}
btns = {}

port = 8501
save_name = None

#### style

st.markdown(
    """
    <style type="text/css">
    section.main > div.block-container {
        padding: 60px 90px;
    }
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:first-child button {
        width: 100%;
        border-color: #FF671D;
        height: 70px;
    }
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:first-child button:hover {
        background-color: #FF671D;
        color: rgb(250, 250, 250);
    }

    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button {
        width: 100%;
    }
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button p {
        font-size: 15px;
    }

    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:first-child button[title="View fullscreen"],
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button[title="View fullscreen"],
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:first-child button[title="Exit fullscreen"],
    section.main > div > div:first-child div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child(2) button[title="Exit fullscreen"] {
        height: 30px;
        width: 30px;
        float: right;
        right: 5px;
        top: 5px;
    }

    section[data-testid="stSidebar"] > div > div:nth-child(2) {
        padding-top: 24px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

##### UI

col00, col01, col02 = st.columns([1, 2, 3])
with col00:
    args["product_prompt"] = st.text_area(label="描述商品（英文）", 
        value=st.session_state.random_prompt if "random_prompt" in st.session_state else "",
        placeholder="请用文字描述商品，比如：a set of cosmetics", height=100)
    
    args["background_prompt"] = st.text_area(label="描述背景（英文）", 
        value=st.session_state.random_prompt if "random_prompt" in st.session_state else "",
        placeholder="请用文字描述背景，比如：beach", height=100)
    
with col01:
    product_images = col01.file_uploader(
        "请上传带有清晰商品的图片，当前只支持正方形比例", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )
    
    if product_images:
        col01.image(
            product_images, 
            use_column_width=True,
        )
        args["product_images"] = streamlit_image2base64(product_images)
        
with col02:
    btns["start_generation"] = st.button("生成", on_click=None)
    
if btns["start_generation"]:
    save_name = str(int(time.time())) + "_" + uuid.uuid4().hex
    data = {
        "product_prompt": args["product_prompt"],
        "background_prompt": args["background_prompt"],
        "height": 1024,
        "width": 1024,
        "save_name": save_name,
        "product_images": args["product_images"],
    }
    
    response = requests.post(f"http://0.0.0.0:{port}/productKV", json=data)
    
if os.path.exists(f'./demos/output/{save_name}.png'):
    print(save_name)
    col02.image(f'./demos/output/{save_name}.png')
        