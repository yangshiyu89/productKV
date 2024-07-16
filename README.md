# 安装
```bash
pip install -r requirements.txt
pip install -v -e .
```
# 模型地址
## sd base: 
https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/tree/main
## unet adapter: 
https://hf-mirror.com/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors
## salient detection
https://hf-mirror.com/briaai/RMBG-1.4


# 测试
更改 ```tests/run_simple_productKV.py``` 中对应模型位置
```bash
python tests/run_simple_productKV.py
```

# 运行 demo
## 启动服务
更改 ```demos/server.py``` 中对应模型位置
```bash
python demos/server.py
```
## 启动 UI 界面
```bash
streamlit run demos/ui.py
```
