FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 中文字体：容器里没有微软雅黑，plot_bar 会回退到 Noto Sans CJK SC，否则中文全是方框
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 embedding 模型：webui 启动时走 HF 离线模式，镜像内必须已有缓存
# 国内构建用 hf-mirror 镜像源，huggingface.co 常不可达
RUN HF_ENDPOINT=https://hf-mirror.com python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

COPY . .

EXPOSE 7860

# .env 不打进镜像，运行时通过 --env-file .env 注入密钥
CMD ["python", "webui.py"]
