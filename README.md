# Streamlit-Multi-image-chat
🚀 A local chatbot that describes images using LLaMA and CLIP models. 🦙
# 🤖 Chat with Image locally

首先，确保您的环境中安装了必要的Python包。这里是一个推荐的环境配置：

bash:
    conda create -n imageChat-env python=3.11
    conda activate imageChat-env
    pip install -r requirements.txt  

model path:
    在代码的 `PRESET_CLIP_MODELS` 和 `PRESET_LLM_MODELS` 中填入需要使用的模型

Usage:
    1.运行脚本文件：
        在终端中执行 `streamlit run imageChat_demo.py`
    2.在浏览器中打开显示的本地URL，即可开始与图像进行交互。
    3.关闭脚本文件：
        在终端中按下 `Ctrl + C`
    4.每次切换模型或者更改模型参数，会重新初始化模型，然后`PRESET_CLIP_MODELS`在内存中无法被释放，所以建议不要频繁切换模型或者更改模型参数