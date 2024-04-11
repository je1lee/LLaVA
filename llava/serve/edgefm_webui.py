import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import requests
from PIL import Image
from io import BytesIO
from transformers import TextIteratorStreamer
import numpy as np
from typing import Iterator

from threading import Thread


class VLMWebUI:
    def __init__(self, args) -> Iterator[str]:
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conversation = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conversation.roles
        
        self.roles = roles
        self.args = args

    @torch.inference_mode
    def inference_from_streaming_frame(self, image: Image.Image, input_text: str) -> str:
        
        # TODO: support multi-turn conversation with the single frame 
        conversation = conv_templates[args.conv_mode].copy()
        
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # first message
        if self.model.config.mm_use_im_start_end:
            input_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_text
        else:
            input_text = DEFAULT_IMAGE_TOKEN + '\n' + input_text
        conversation.append_message(conversation.roles[0], input_text)
        image = None
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs=input_ids,
            streamer=streamer,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

def main(args):
    print(args)
    vlm_webui = VLMWebUI(args)
    
    images = [
        "samples/image1.png",
        "samples/image2.png",
        "samples/image3.png",
        "samples/image4.png",
        "samples/image5.png",
    ]
    
    prompts = [
        "What happens in this scene?",
        "Please answer in json format: \{ Car: boolean \}",
        "How many people in this scene?",
        "Is there any apple?",
        "Describe the scene in the image",
        "What is the background color of this image?"
    ]
    
    iteration_test = 100
    for idx in range(iteration_test):
        
        image = images[idx % len(images)]
        prompt = prompts[idx % len(prompts)]
        
        output_iterator = vlm_webui.inference_from_streaming_frame(Image.open(image).convert('RGB'), prompt)
        
        for chunk in output_iterator:
            print(chunk, flush=True, end="")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)