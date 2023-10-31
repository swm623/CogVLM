# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import random

from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL


class VAE(nn.Module):
    def __init__(self,
                 vae_path='/nfs/users/zhangsan/models/stable-diffusion-xl-base-1.0',
                 max_batch_size=2,
                 device=torch.device('cuda:0'),
                 dtype=torch.bfloat16,
                 compile=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        self.max_batch_size = max_batch_size
        self.vae.requires_grad_(False)

        if compile:
            self.vae = torch.compile(self.vae)  # requires PyTorch 2.0
        self.vae.to(device, dtype=dtype)  # VAE must run on float32

    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(list(x))
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(memory_format=torch.contiguous_format).float()
        x = x.to(self.device, dtype=self.vae.dtype)

        latents = []
        for i in range(0, x.shape[0], self.max_batch_size):
            _latent = self.vae.encode(x[i:i + self.max_batch_size, ...]).latent_dist.sample()
            latents.append(_latent)
        latent = torch.cat(latents, dim=0)
        latent = latent.float() * self.vae.config.scaling_factor
        return latent


class TextEncoder(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path='/nfs/users/zhangsan/models/stable-diffusion-xl-base-1.0',
                 revision=None,
                 max_batch_size=2,
                 device=torch.device('cuda:0'),
                 dtype=torch.float16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        # "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
        self.proportion_empty_prompts = 0

        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer",
                                                      revision=revision, use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2",
                                                      revision=revision, use_fast=False)

        # import correct text encoder classes
        text_encoder_cls_one = self.import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision,
                                                                               subfolder="text_encoder")
        text_encoder_cls_two = self.import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision,
                                                                               subfolder="text_encoder_2")

        text_encoder_one = text_encoder_cls_one.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                                revision=revision)
        text_encoder_two = text_encoder_cls_two.from_pretrained(pretrained_model_name_or_path,
                                                                subfolder="text_encoder_2", revision=revision)
        text_encoder_one.to(self.device, dtype=self.dtype)
        text_encoder_two.to(self.device, dtype=self.dtype)

        self.text_encoders = [text_encoder_one, text_encoder_two]
        self.tokenizers = [tokenizer_one, tokenizer_two]

    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path, revision, subfolder):
        text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder,
                                                               revision=revision)

        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")

    def forward_slice(self, model, x, **kwargs):
        y = []
        for i in range(0, x.shape[0], self.max_batch_size):
            _y = model(x[i:i + self.max_batch_size, ...], **kwargs)
            y.append(_y)
        y = torch.cat(y, dim=0)
        return y

    @torch.no_grad()
    def forward(self, prompt_batch, is_train=True):
        prompt_embeds_list = []

        captions = []
        for caption in prompt_batch:
            if self.proportion_empty_prompts > 0 and random.random() < self.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption)
                                if is_train else caption[0])

        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            prompt_embeds = text_encoder( text_input_ids, output_hidden_states=True)
            # prompt_embeds = self.forward_batch(text_encoder, text_input_ids, output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}
