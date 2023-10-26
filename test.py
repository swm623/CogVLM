# -*- encoding: utf-8 -*-
import os, sys, glob
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

def main():
    prompt = """
Describe this image

Here are a few things to keep in mind when describing the content of an image in detail:
Provide enough contextual information for the model to understand the key features of the image.
Use clear, concise language to describe the image and avoid lengthy and unnecessary descriptions.
Describe the elements in the image in a logical and spatial order so that readers can easily follow the description to understand the image.
Be sure to stick to the 70-word limit to keep descriptions concise.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    directory_path = "./laion-high/"

    # 使用glob.glob获取匹配的文件列表
    jpg_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))    
    jpg_dict = {}
    txt_dict = {}
    result_list = []
    for  value in jpg_files:
        parts = value.split("/")
        file_name_without_extension = parts[2].split(".")[0]
        jpg_dict[file_name_without_extension] = value
    for  value in txt_files:
        parts = value.split("/")
        file_name_without_extension = parts[2].split(".")[0]
        
        with open(value, 'r') as file:
            # 使用 read() 方法读取整个文件内容
            file_content = file.read()
            txt_dict[file_name_without_extension] = file_content

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    count = 0 
    with torch.no_grad():
        for key, value in jpg_dict.items():
            try:
                response, history, cache_image = chat(
                    value, 
                    model, 
                    text_processor_infer,
                    image_processor,
                    prompt, 
                    history=None, 
                    image=None, 
                    max_length=args.max_length, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    no_prompt=args.no_prompt
                    )
            except Exception as e:
                print(e)
                break
            info_dict = {"name":key,"cogvlm":response,"org":txt_dict[key]}
            result_list.append(info_dict)
            count += 1
            if count % 10 == 0 :
                file_path = "data1.json"

                # 使用 json.dump() 函数将数据写入文件
                with open(file_path, 'w') as json_file:
                    json.dump(result_list, json_file)                
    
    file_path = "data1.json"

    # 使用 json.dump() 函数将数据写入文件
    with open(file_path, 'w') as json_file:
        json.dump(result_list, json_file)
if __name__ == "__main__":
    main()
