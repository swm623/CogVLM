import webdataset as wds
import braceexpand
#from aesthetic import aesthetic
# -*- encoding: utf-8 -*-
import os, sys, glob
import traceback
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor
from sat.helpers import print_all
import logging
class LaionDataset:
    def __init__(self) -> None:
        self.prompt = """
    Describe this image
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
        parser.add_argument("--self_world_size", type=int, default=1, help='world_size')      
        parser.add_argument("--self_rank", type=int, default=0, help='rank')       
        parser.add_argument("--self_local_rank", type=int, default=0, help='local_rank')     
        parser.add_argument("--source_path", type=str, default='/ML-A100/sshare-app/yanjinbing/laion-high-aesthetics_6', help='source_path')  
        parser.add_argument("--target_path", type=str, default='/ML-A100/sshare-app/swmall/data/test/laion-high', help='target_path')
        parser.add_argument("--run_mode", type=str, default='python', help='run_mode')        
        self.args = parser.parse_args()
        parser = CogVLMModel.add_model_specific_args(parser)
        self.args = parser.parse_args()

        print_all(self.args)
        self.source_path = self.args.source_path
        self.target_path = self.args.target_path
        self.max_length = self.args.max_length
        self.top_p = self.args.top_p
        self.top_k = self.args.top_k
        self.temperature = self.args.temperature
        self.no_prompt = None
        rank = int(os.environ.get('RANK', 0))
        local_rank =  int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.self_local_rank = self.args.self_local_rank
        self.self_rank = self.args.self_rank
        self.self_world_size = self.args.self_world_size            
        if self.args.run_mode != "python":
            self.self_local_rank = local_rank
            self.self_world_size  = world_size
            self.self_rank = rank

        print_all(f"self_local_rank {self.self_local_rank}")
        print_all(f"self_world_size {self.self_world_size}")    
        print_all(f"self_rank {self.self_rank}")                     
        #torch.cuda.set_device(rank)
        #self.device = torch.device('cuda:{:d}'.format(self_local_rank))
        print_all(f"current_device {torch.cuda.current_device()}")
        # load model
        self.model, self.model_args = CogVLMModel.from_pretrained(
            self.args.from_pretrained,
            args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device='cuda',
            **vars(self.args)
        ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
        self.model = self.model.eval()
        from sat.mpu import get_model_parallel_world_size
        assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        self.tokenizer = llama2_tokenizer(self.args.local_tokenizer, signal_type='chat')
        self.image_processor = get_image_processor(self.model_args.eva_args["image_size"][0])

        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        self.text_processor_infer = llama2_text_processor_inference(self.tokenizer, self.max_length, self.model.image_length)
    
    def make_dataset(self, no):
        start = time.time()
        s_no = f'{no:0>3}'
        d_no = s_no
        source_files = os.path.join(self.source_path ,  s_no +".tar")
        target_file = os.path.join(self.target_path , d_no + ".tar")
        count = 0
        print_all(f"{source_files} --> {target_file}")
        #dataset = wds.WebDataset(source_files).decode('pil')

        pipeline = [
            wds.SimpleShardList(source_files),
            wds.tarfile_to_samples(),
            wds.decode('pil')
        ]
        dataset = wds.DataPipeline(*pipeline)
        with wds.TarWriter(target_file) as out_stream:
            for sample in dataset:
                jpg = sample['jpg']
                try:
                    response, history, cache_image = chat(
                        None, 
                        self.model, 
                        self.text_processor_infer,
                        self.image_processor,
                        self.prompt, 
                        history=None, 
                        image=None, 
                        max_length=self.max_length, 
                        top_p=self.top_p, 
                        temperature=self.temperature,
                        top_k=self.top_k,
                        invalid_slices=self.text_processor_infer.invalid_slices,
                        no_prompt=self.no_prompt,
                        force_pil_image = jpg
                        )
                except Exception as e:
                    traceback.print_exc()
                    break
                sample['json']['recaption'] = response
                out_stream.write(sample)
                if count % 50 == 0:
                    end = time.time()
                    usetime = end - start
                    start = time.time()
                    print_all(f"target_file {target_file}, count {count},执行时间: {usetime}")
                count += 1

        return count
    def run(self):
        total_data = 100
        rank = self.self_rank
        world_size = self.self_world_size     
        print_all(f"rank {rank}  world_size {world_size}")
        # 计算每个进程的数据范围
        data_per_process = total_data // world_size  # 每个进程平均处理的数据量
        remainder = total_data % world_size  # 余下的数据

        # 计算当前进程应该处理的数据范围
        start_index = rank * data_per_process  # 起始索引
        end_index = start_index + data_per_process  # 结束索引
        # 如果有余下的数据，分配给前几个进程
        if rank < remainder:
            start_index += rank
            end_index += rank + 1
        else:
            start_index += remainder
            end_index += remainder
        print_all(f"start_index {start_index}  end_index {end_index}")
        total = 0
        for i in range(start_index, end_index):
            start = time.time()
            print_all(f"start {i} ")
            count = self.make_dataset(i)
            end = time.time()
            usetime = end - start
            total += count
            print_all(f"end {i} , total is {total}, cur file count is {count} 执行时间: {usetime}")

laionDataset = LaionDataset()

if __name__ == "__main__":
    print_all(__file__)
    laionDataset.run()