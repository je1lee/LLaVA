FROM dregistry.nota.ai/nota/people/hyoungkyu.song/nvidia/pytorch:23.09-py3

RUN mkdir -p /workspace
WORKDIR /workspace

COPY . /workspace
RUN pip install -e . && \
    pip uninstall -y transformer-engine && \
    rm -rf /root/.cache/pip

ENV HF_HOME=/CHECKPOINT/huggingface-cache-dir