## Installation

### Docker image build

```bash
export TAG=v$(cat ./VERSION) && docker build -t edgefm-llava-torch:${TAG} .
```

### Docker compose

You may copy the [`docker-compose-torch.yml`](./docker-compose-torch.yml) and modify with your environment.
Here is the example `docker-compose.yml` customized for the specific environment.

<details><summary>Example yaml file</summary>

```yaml
version: "3.9"

# export TAG=v$(cat ./VERSION) && docker compose -f docker-compose-private.yml run --service-ports --name edgefm-llava-torch edgefm-llava-torch bash

services:
  edgefm-llava-torch:
    build:
      context: .
      dockerfile: Dockerfile
    image: dregistry.nota.ai/nota/people/hyoungkyu.song/edgefm-llava-torch:${TAG}
    container_name: edgefm-llava-torch-${TAG}
    ipc: host
    ports:
      - "50002:50002" # (optional, gradio) configuration helper
      - "50003:50003" # (optional, gradio) inference demo
    volumes:
      # from path: your working directory
      - /home/hyoungkyu.song/edgefm-llava:/workspace
      # from path: your dataset directory
      - /home/hyoungkyu.song/DATA:/DATA
      # from path: your checkpoint directory
      - /home/hyoungkyu.song/CHECKPOINT:/CHECKPOINT
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["4", "5", "6", "7"] # your GPU id(s)
              capabilities: [gpu]

```

</details>

<br/>

After then, you could execute with the following command:

```bash
export TAG=v$(cat ./VERSION) && docker compose -f docker-compose-private.yml run --service-ports --name edgefm-llava-torch edgefm-llava-torch bash
```

## Usage

### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

## FAQ

### Error in transformer_engine with the following message: `.../transformer_engine_extensions.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7...`

```bash
pip uninstall transformer-engine
```