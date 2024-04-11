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