import json
import random
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image


class ImageProcess():
    def __init__(self, resolution=256,
                 random_crop=False, dump_image=False):
        self.resolution = resolution
        self.random_crop = random_crop
        self.dtype = torch.float16
        self.dump_image = dump_image
        self.dump_index = 0

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def resize(self, image, shortL):
        H, W, C = image.shape
        if H > W:
            dst_size = (shortL, round(H * shortL / W))  # (width,height)
        else:
            dst_size = (round(W * shortL / H), shortL)

        I = cv2.resize(image, dst_size, cv2.INTER_LINEAR)
        return I

    def crop(self, image, L, random=False):
        H, W, C = image.shape
        if random:
            x = torch.randint(0, W - L + 1, size=(1,)).item()
            y = torch.randint(0, H - L + 1, size=(1,)).item()
        else:
            x = max(0, int(round((W - L) / 2.0)))
            y = max(0, int(round((H - L) / 2.0)))

        crop_top_left = torch.tensor((y, x), dtype=torch.long)
        I = image[y:y + L, x:x + L, :]
        return I, crop_top_left

    def normalize(self, x):
        x = torch.from_numpy(x)  # torch.uint8
        x = (x - 127.5) * (1 / 127.5)  # torch.float32
        return x

    def process(self, image_src, **kwarg):
        # bucket is dict:  {"idx": idx,  "size": (h, w) }
        # where (h, w) is 64 stepped and w * h is about 1024*1024
        L = self.resolution
        def_size ={"size": (L, L)}
    
        bucket = kwarg.get('bucket', def_size)

        if bucket is None:
            bucket = def_size

        if image_src.dtype == np.float32:
            image_src = (image_src * 255).astype(np.uint8)


        H, W, C = image_src.shape
        # print(image_src.shape, image_src.dtype)
        original_size = torch.tensor((H, W), dtype=torch.long)

        image = self.resize(image_src, self.resolution)
        image, crop_top_left = self.crop(image, self.resolution, random=self.random_crop)

        # save to disk
        if self.dump_image:
            self.save_image(image, f"{H}x{W}-{crop_top_left[0]}-{crop_top_left[1]}-{L}", **kwarg)

        image = self.normalize(image)

        size_of_original_crop_target = torch.cat([original_size,
                                                  crop_top_left,
                                                  torch.tensor((bucket['size']), dtype=torch.long)])

        return image, size_of_original_crop_target

    def save_image(self, image, posfix, **kwarg):
        if 'key' in kwarg:
            key = f"img_{kwarg['key']}"
        else:
            key = f"img_{self.dump_index}"
            self.dump_index += 1

        img = Image.fromarray(image, "RGB")
        img.save(f"output/{key}_{posfix}.jpg")
