import streamlit as st
from PIL import Image
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import tempfile
import os
import asyncio
import gc

# Set page configuration
st.set_page_config(
    page_title="MiniCPM Streamlit",
    page_icon=":robot:",
)

# 预设CLIP模型列表，每个元素是一个字典，包含模型名称和对应的描述等信息
PRESET_CLIP_MODELS = [
    {"name": "mmproj-model-f16.gguf", "path": "./mmproj-model-f16.gguf"},
]

# 预设llm模型列表，每个元素是一个字典，包含模型名称和对应的描述等信息
PRESET_LLM_MODELS = [
    {
        "name": "Model-7.6B-Q4_0_4_4.gguf",
        "path": "./Model-7.6B-Q4_0_4_4.gguf"
    },
    {
        "name": "Qwen2.5-7B-Instruct-Q4_0_4_4.gguf",
        "path": "./Qwen2.5-7B-Instruct-Q4_0_4_4.gguf"
    }
]

# 缩小图片尺寸的操作实际上是将图片进行了缩放，而不是裁剪。具体来说，使用 img.thumbnail(max_size) 方法会将图片缩放到不超过指定的最大宽度和高度，同时保持原始图片的宽高比。
def image_to_base64_data_uri(file_path, max_size=(400, 400)):
    with Image.open(file_path) as img:
        img.thumbnail(max_size)  # 缩放图片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            img.save(temp_img, format="PNG")
            with open(temp_img.name, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"

# Function to handle file upload
async def handle_file_upload(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# 初始化会话状态
if 'llm_image' not in st.session_state:
    st.session_state.llm_image = None
if 'selected_clip_model' not in st.session_state:
    st.session_state.selected_clip_model = None
if 'selected_llm_model' not in st.session_state:
    st.session_state.selected_llm_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generation_config' not in st.session_state:
    st.session_state.generation_config = None
if 'n_thread' not in st.session_state:
    st.session_state.n_thread = 8
if 'n_ctx' not in st.session_state:
    st.session_state.n_ctx = 2048
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 512
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.8
if 'top_k' not in st.session_state:
    st.session_state.top_k = 100



@st.cache_resource
def init_llm_image(generation_config):
    chat_handler = MiniCPMv26ChatHandler(clip_model_path=generation_config.get('selected_clip_model'))
    llm_image = Llama(
        model_path=generation_config.get('selected_llm_model'),
        chat_handler=chat_handler,
        chat_format="minicpm-v-2.6",
        n_ctx=generation_config['n_ctx'],
        n_threads=generation_config.get('n_threads'),
    )
    return llm_image

def render_generation_config_ui():
    with st.sidebar:
        st.sidebar.header("Select models")
        selected_clip_model = st.selectbox("选择CLIP模型", [model["path"] for model in PRESET_CLIP_MODELS], key="clip_model_select")
        selected_llm_model = st.selectbox("选择语言模型", [model["path"] for model in PRESET_LLM_MODELS], key="llm_model_select")
        st.sidebar.header("Parameters")
        n_thread = st.slider("N_thread", 1, 12, 8, step=1, key="n_thread_slider")
        n_ctx = st.slider("N_ctx", 0, 4096, 2048, step=2, key="n_ctx_slider")
        max_tokens = st.slider("Max_tokens", 0, 1024, 512, step=1, key="max_tokens_slider")
        top_p = st.slider("Top_p", 0.0, 1.0, 0.8, step=0.01, key="top_p_slider")
        top_k = st.slider("Top_k", 0, 100, 100, step=1, key="top_k_slider")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01, key="temperature_slider")
        st.session_state.selected_clip_model = selected_clip_model
        st.session_state.selected_llm_model = selected_llm_model
        st.session_state.n_thread = n_thread
        st.session_state.n_ctx = n_ctx
        st.session_state.max_tokens = max_tokens
        st.session_state.top_p = top_p
        st.session_state.top_k = top_k
        st.session_state.temperature = temperature

def get_generation_config():
    selected_clip_model = st.session_state.selected_clip_model
    selected_llm_model = st.session_state.selected_llm_model
    n_thread = st.session_state.n_thread
    n_ctx = st.session_state.n_ctx
    max_tokens = st.session_state.max_tokens
    top_p = st.session_state.top_p
    top_k = st.session_state.top_k
    temperature = st.session_state.temperature
    generation_config = {
        'n_thread': n_thread,
        'n_ctx': n_ctx,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'selected_clip_model': selected_clip_model,
        'selected_llm_model': selected_llm_model,
    }
    return generation_config

# 侧边栏UI布局如下：
st.sidebar.header("Image Multimodel Chat")
uploaded_image = st.sidebar.file_uploader("Upload image", key=1, type=["jpg", "jpeg", "png"],
                                            accept_multiple_files=False)
image_path = asyncio.run(handle_file_upload(uploaded_image))
if uploaded_image is not None:
    if not any(msg['image'] == uploaded_image for msg in st.session_state.chat_history):            
        # 将上传的图像添加到聊天历史中
        st.session_state.chat_history.append({"role": "user", "content": None, "image": uploaded_image})

buttonClean = st.sidebar.button("Clear chat history", key="clean")
if buttonClean:
    st.session_state.chat_history = []
    st.session_state.response = ""
    st.rerun()

# 渲染参数配置 UI
render_generation_config_ui()

# 其他页面内容
st.title("💻 Local Image Description Bot 🤖")
st.caption("🚀 A chatbot that describes images using LLaMA and CLIP models. 🦙")



# 初始化模型
if st.session_state.generation_config is None:
    st.session_state.generation_config = get_generation_config()
    st.session_state.llm_image = init_llm_image(st.session_state.generation_config)
    llm = st.session_state.llm_image
elif st.session_state.llm_image is None:
    st.session_state.llm_image = init_llm_image(st.session_state.generation_config)
    llm = st.session_state.llm_image
else:
    llm = st.session_state.llm_image
# 检查参数变化进而重新初始化模型
if st.session_state.generation_config != get_generation_config():
    st.session_state.generation_config = get_generation_config()
    if st.session_state.llm_image is not None:
        del st.session_state.llm_image
        st.cache_resource.clear()
        gc.collect()  # 强制垃圾回收
    st.session_state.llm_image = init_llm_image(st.session_state.generation_config)
    llm = st.session_state.llm_image

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            if message["image"] is not None:
                st.image(message["image"], width=300, use_column_width=False)
            elif message["content"] is not None:
                st.markdown(message["content"])
    else:
        with st.chat_message(name="model", avatar="assistant"):
            st.markdown(message["content"])

# User input box
user_text = st.chat_input("Enter your question", key="user_input")
if user_text:
    with st.chat_message(U_NAME, avatar="user"):
        st.session_state.chat_history.append({"role": "user", "content": user_text, "image": None})
        st.markdown(f"{U_NAME}: {user_text}")

    with st.chat_message(A_NAME, avatar="assistant"):
        # If the previous message contains an image, pass the image to the model
        # 确保图片上传了
        if image_path is not None:
            data_uri = image_to_base64_data_uri(image_path)
        else:
            imagefile = None
            data_uri = None
        # 创建消息列表
        msgs = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}} if data_uri else None,
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        # 过滤掉 None 值
        msgs = [msg for msg in msgs if msg["content"] is not None]
        # 创建聊天完成请求
        response = llm.create_chat_completion(
            messages=msgs,
            stream=True,
            max_tokens=st.session_state.generation_config['max_tokens'],
            top_p=st.session_state.generation_config['top_p'],  # 采样概率阈值
            top_k=st.session_state.generation_config['top_k'],  # 采样数量
            temperature=st.session_state.generation_config['temperature']  # 温度
        )
        try:
            # 创建一个生成器函数，用于提取和处理每个 chunk 中的 content
            def extract_content(response):
                generated_text = ""
                for chunk in response:
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # 去除换行符
                            # content = content.replace('\n', ' ')
                            generated_text += content
                            yield content
                return generated_text

            # 使用 st.write_stream 进行流式输出
            generated_text = st.write_stream(extract_content(response))
            # 将生成的完整文本添加到会话历史记录中
            st.session_state.chat_history.append({"role": "model", "content": generated_text, "image": None})
        except Exception as e:
            st.error(f"Error generating response: {e}")
        st.divider()