# coding=utf-8

from typing import Union, Optional
import logging
import os
import io
import zlib
from collections import Counter
from tqdm import tqdm
from PIL import Image
import webdataset as wds
from module.nn import TextEncoder

import torch
from torch import multiprocessing

from typing import Optional, Union

logger = logging.getLogger(__name__)

g_finish_num = None
g_text_encoder = None


class WebdatasetConverter():
    def __init__(self, src_path: str, dst_path: str,
                 pretrained_model_name_or_path: str,
                 batch_size=1024, max_tar_num=None,
                 min_image_size: int = None, num_proc=0, num_gpus=8,
                 text_column: Optional[Union[tuple, str]] = ('json', 'caption'),
                 drop_columns: Optional[Union[tuple, str]]=None,
                 recover_mode=True,
                 use_npz=False
                 ) -> None:
        """
            text_column:  tuple|str
            recover_mode:  recover from last run. existed item in dst_path will skipped
        """
        self.src_path = src_path
        self.dst_path = dst_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.min_image_size = min_image_size
        self.num_proc = num_proc
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.max_tar_num = max_tar_num
        self.text_column = text_column
        self.drop_columns = drop_columns
        self.datafile_num = 0
        self.recover_mode = recover_mode
        self.use_npz = use_npz
        pass

    def init_encoder(self, dev_no):
        global g_text_encoder
        cuda_dev = f"cuda:{dev_no}"
        torch.cuda.set_device(dev_no)
        g_text_encoder = TextEncoder(
            self.pretrained_model_name_or_path,
            dtype=torch.float16,
            device=torch.device(cuda_dev)
        )

    def _subprocess_init(self, no: multiprocessing.Value, finish_num):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)-15s %(name)s %(levelname)s: %(message)s")

        global g_finish_num
        g_finish_num = finish_num

        with no.get_lock():
            proc_id = no.value
            no.value += 1

        dev_no = proc_id % self.num_gpus
        print(f"[{proc_id}]: cuda:{dev_no} gpus={self.num_gpus}")

        self.init_encoder(dev_no)

    def convert(self):
        data_files = []
        idx = 0
        exist_num = 0
        for dir, _, files in os.walk(self.src_path):
            for fn in files:
                if not fn.endswith(".tar"):
                    continue
                if self.recover_mode:
                    # skip file those dstfile exists and is larger than 1GB
                    dst_file = os.path.join(self.dst_path, fn)
                    if os.path.exists(dst_file):
                        st = os.stat(dst_file)
                        if st.st_size > 1024 * 1024 * 1024:
                            # skip
                            exist_num += 1
                            continue
                idx += 1
                data_files.append(os.path.join(dir, fn))
                if self.max_tar_num is not None and len(data_files) >= self.max_tar_num:
                    break

        self.datafile_num = len(data_files)

        if self.datafile_num == 0:
            logger.info(f"tar file not found in {self.src_path}")
            return

        data_files = [(idx, fn) for idx, fn in enumerate(sorted(data_files))]
        logger.info(
            f"converting {self.datafile_num} to {self.dst_path}. {exist_num} exist files skipped")

        counter = Counter()

        if self.num_proc <= 1:
            # user only one gpus
            global g_finish_num
            g_finish_num = 0

            self.init_encoder(0)
            for fn in tqdm(data_files):
                counter.update(self.convert_single(fn))
        else:
            no = multiprocessing.Value('i', 0)
            finish_num = multiprocessing.Value('i', 0)

            self.num_proc = min(self.num_proc, self.datafile_num)
            pool = multiprocessing.Pool(
                self.num_proc, initializer=self._subprocess_init, initargs=(no, finish_num))
            counters = pool.map(self.convert_single, data_files)
            pool.close()
            pool.join()

            for c in counters:
                counter.update(c)

        logger.info(f"{counter.most_common()}")

    def convert_single(self, info):
        global g_text_encoder, g_finish_num
        assert g_text_encoder is not None and g_finish_num is not None

        idx, fullname = info

        torch.cuda.empty_cache()

        counter = Counter()
        basename = os.path.basename(fullname)
        out_file = os.path.join(self.dst_path, basename)
        with wds.TarWriter(out_file) as out_stream:
            # logger.info(f"processing {src_file}")
            # txt, json, jpg:png
            src_ds = wds.WebDataset(fullname).decode()
            if isinstance(g_finish_num, int):
                num = g_finish_num
            else:
                num = g_finish_num.value
            desc = f"[{num}/{idx}/{self.datafile_num}]: {basename}"
            batch = []
            for item in tqdm(src_ds, desc=desc):
                batch.append(item)
                if len(batch) == self.batch_size:
                    batch = self.batch_encode(batch, g_text_encoder, counter)
                    for sample in batch:
                        out_stream.write(sample)
                    batch = []
            if batch:
                batch = self.batch_encode(batch, g_text_encoder, counter)
                for sample in batch:
                    out_stream.write(sample)
        if isinstance(g_finish_num, int):
            g_finish_num += 1
        else:
            g_finish_num.value += 1
        return counter

    def verify_image_data(self, sample: dict):
        image_src = None
        for k, v in sample.items():
            if k in ['jpg', 'png']:
                image_src = v
                break

        if not image_src:
            return False, "jpg/png not found"

        try:
            image = Image.open(io.BytesIO(image_src))
            if self.min_image_size and (image.width < self.min_image_size or image.height < self.min_image_size):
                return False, f"size{image.width, image.height} is smaller than {self.min_image_size}"

            # try load image data
            image.load()
        except Exception as e:
            return False, f"error: {e}"

        return True, ""

    def batch_encode(self, batch, text_encoder, counter: Counter):
        # check image-data
        # do text encoding
        col = self.text_column
        if isinstance(col, (tuple, list)):
            texts = [sample[col[0]][col[1]] for sample in batch]
        elif col in sample:
            texts = [sample[col] for sample in batch]
        encoded = text_encoder.forward(texts)
        result = []
        for i, sample in enumerate(batch):
            ret, reason = self.verify_image_data(sample)
            if not ret:
                if reason.startswith("error"):
                    logger.warning(f"{sample['__key__']} dropped: {reason}")
                continue
            if encoded:
                if self.use_npz:
                    sample['npz'] = {}
                    for k, v in encoded.items():
                        sample['npz'][k] = v[i].squeeze(0).detach().numpy()
                else:
                    for k, v in encoded.items():
                        sample[k+".npy"] = v[i].squeeze(0).detach().numpy()
            if self.drop_columns:
                if isinstance(self.drop_columns, str):
                    sample.pop(self.drop_columns)
                else:
                    for c in self.drop_columns:
                        if c in sample:
                            sample.pop(c)
                result.append(sample)

        if counter is not None:
            counter.update({"total": len(batch), "converted": len(result)})
        return result
