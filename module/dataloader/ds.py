import os
import numpy as np
import torch
import torch.utils.data
import pickle
from .image_process import ImageProcess


def loadFromFile(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.dtype = torch.float16
        self.rank = kwargs.get('rank', None)
        self.training = kwargs.get('training', True)
        self.verbose = kwargs.get('verbose', False)
        self.image_size = kwargs.get('image_size', False)
        self.use_vae_encode_online = kwargs.get('use_vae_encode_online', False)
        self.random_crop = kwargs.get('random_crop', False)

        self.image_process = ImageProcess(self.image_size, random_crop=self.random_crop)

        self.datafile = kwargs.get('train_datafile' if self.training else 'test_datafile', None)
        if self.verbose:
            print(f'[Dataset for Unet] datafile: {self.datafile}')

        self.data = loadFromFile(self.datafile)
        self.N = len(self.data)
        if self.verbose:
            print('[rank-%d] len_train_dataset=%d' % (self.rank, self.N))

    def __len__(self):
        return self.N * 100

    def __getitem__(self, index):
        index = index % self.N
        data = self.data[index]

        if self.use_vae_encode_online:
            image, size_of_original_crop_target = self.image_process(data['image_src'])
            ret = {'image': image,
                   'size_of_original_crop_target': size_of_original_crop_target}
        else:
            ret = {'latent': data['latent'],
                   'size_of_original_crop_target': data['size_of_original_crop_target']}

        ret['prompt_embeds'] = data['prompt_embeds']
        ret['pooled_prompt_embeds'] = data['pooled_prompt_embeds']
        return ret


