import webdataset as wds
import braceexpand
import logging
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

logger = logging.getLogger("__name__")
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

class LaionDataset:
    source_path = '/ML-A100/sshare-app/yanjinbing/laion-high-aesthetics_6'
    target_path = '/ML-A100/sshare-app/saiwanming/laion-high-resolution-output/'
    def __init__(self) -> None:
        self.prompt = """
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
        parser = CogVLMModel.add_model_specific_args(parser)
        args = parser.parse_args()
        print(args)
        self.max_length = 2048
        self.top_p = 0.4
        self.top_k = 1
        self.temperature = 0.8
        self.no_prompt = None
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        # load model
        """
            

        self.model, self.model_args = CogVLMModel.from_pretrained(
            args.from_pretrained,
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
            **vars(args)
        ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
        self.model = self.model.eval()
        from sat.mpu import get_model_parallel_world_size
        assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

        self.tokenizer = llama2_tokenizer('lmsys/vicuna-7b-v1.5', signal_type='chat')
        self.image_processor = get_image_processor(self.model_args.eva_args["image_size"][0])

        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

        self.text_processor_infer = llama2_text_processor_inference(self.tokenizer,  args.max_length, self.model.image_length)
        """
    def make_dataset(self, dataset):
        start = time.time()
        s_no = f'{1:0>3}'
        d_no = s_no
        target_file = os.path.join(self.target_path , d_no + ".tar")
        count = 0
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
                if count % 50 == 0:
                    end = time.time()
                    usetime = end - start
                    start = time.time()
                    print(f"target_file {target_file}, count {count},执行时间: {usetime}")
                count += 1

        return count
    def run(self):
        source_file = os.path.join(self.source_path ,  "{000..099}.tar")

        urls = list(braceexpand.braceexpand(source_file))
        pipeline = [
            wds.SimpleShardList(urls)
        ]
        pipeline.extend([
            wds.shardlists.split_by_node,
            wds.shardlists.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pil", handler=log_and_continue),
        ])
        dataset = wds.DataPipeline(*pipeline)
        self.make_dataset(dataset)

laionDataset = LaionDataset()

if __name__ == "__main__":
    print(__file__)
    laionDataset.run()