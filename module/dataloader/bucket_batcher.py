# coding=utf-8

from collections import Counter
import logging
from random import randint
import webdataset as wds

logger = logging.getLogger(__name__)


class AspectRatioBucketBatcher(wds.PipelineStage):
    """
        support sdxl image bucket strategy. https://mdnice.com/writing/682896d9c0e74709a46d85da4fbfecaa
        sample in same batch will be in same bucket, and 'bucket' is added to sample (dict):
        "bucket": {
            "idx": bucket_idx,
            "size": (h, w)
        }

        to use in webdataset, add AspectRatioBucketBatcher to pipeline. 
        input sample should be a dict with key 'size' or 'image'
    """

    def __init__(self, batch_size=2, image_size_base=1024, image_size_step=64, image_key=None, size_key=None, filter_param=None):
        assert (image_key is not None and size_key is None) or (image_key is None and size_key is not None)
        self.image_size_base = image_size_base
        self.image_size_step = image_size_step
        self.batch_size = batch_size
        self.image_key = image_key
        self.size_key = size_key
        self.filter_param = filter_param if filter_param is not None else {}
        self.bucket_table = self._build_bucket_table(
            self.image_size_base, self.image_size_step)

    def run(self, src):
        # fetch samples by bucket.
        for samples, bucket_idx in self._fill_bucket(src):
            for sample in samples:
                sample['bucket'] = {'idx': bucket_idx,
                                    'size': self.bucket_table[bucket_idx]}
                yield sample

    def _calc_size_dist(self, h, w):
        s = w * h - self.image_size_base*self.image_size_base
        return s if s >= 0 else -s

    def _dist_info(self, h, w):
        s = w * h - self.image_size_base*self.image_size_base
        return h, w, s, round(h/w, 3)

    def _find_bucket(self, h, w):
        # todo: use binary search ...
        r = h / w
        for i, b in enumerate(self.bucket_table):
            if r < b[0]/b[1]:
                return i
        return 0

    def _build_bucket_table(self, size, step):
        # bucket_table with ratio [0]/[1] from low to high
        bucket_table = []
        h, w = size//2, size*2
        while h <= size*2 and w >= size//2:
            bucket_table.append((h, w))
            d1, d2 = self._calc_size_dist(h+step, w), self._calc_size_dist(h, w-step)
            if d1 < d2:
                h += step
            else:
                w -= step
        # logger.info(f"bucket_table: {len(bucket_table)} items.  {bucket_table}")
        return bucket_table
    
    def _get_sample_size(self, sample):
        # get size from sample
        h, w = None, None
        if self.image_key is not None:
            if self.image_key in sample:
                h, w = sample[self.image_key].shape[0], sample[self.image_key].shape[1]
            else:
                logger.warning(f"image_key(`{self.image_key}`) not found in sample: {sample.keys()}")
        elif self.size_key is not None:
            if self.size_key in sample:
                h, w = sample[self.size_key]
            else:
                logger.warning(f"size_key(`{self.size_key}`) not found in sample: {sample.keys()}")
        return h, w
    
    def _filter_by_size(self, sample, h, w):
        # check filter
        drop_size = self.filter_param.get('drop_small_image_size', None)
        if drop_size is not None and w < drop_size and h < drop_size:
            logger.info(f"dropping by size: key={sample['__key__']}, size={(w, h)}")
            return True

        drop_ratio = self.filter_param.get('drop_image_aspect_ratio', None)
        if drop_ratio is not None and (w / h > drop_ratio or h / w > drop_ratio):
            logger.info(
                f"dropping by aspect ratio: key={sample['__key__']}, size={(w, h)}, ratio={w/h:.3f}")
            return True

        return False

    def _fill_bucket(self, src):
        sample_buf = [[] for b in self.bucket_table]
        for sample in src:
            # get size from sample
            h, w = self._get_sample_size(sample)
            if h is None or self._filter_by_size(sample, h, w):
                continue

            idx = self._find_bucket(h, w)
            sample_buf[idx].append(sample)
            if len(sample_buf[idx]) == self.batch_size:
                yield sample_buf[idx], idx
                sample_buf[idx] = []

        # combine near-bucket ...
        batch = []
        for idx, item in enumerate(sample_buf):
            batch.extend(item)
            sample_buf[idx] = []
            while len(batch) >= self.batch_size:
                yield batch[0:self.batch_size], idx
                batch = batch[self.batch_size:]
        if len(batch) > 0:
            # there is no more sample in the last batch
            assert len(batch) < self.batch_size
            logger.info(f"dropping {len(batch)} sample in bucket")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    def test_src(total):
        # size in (h, w)
        for i in range(total):
            yield {"size": (randint(256, 2048), randint(256, 2048)), "__key__": i}

    filter_param = {"drop_small_image_size": 512, "drop_image_aspect_ratio": 4.0}

    batch_size = 4
    pipe = AspectRatioBucketBatcher(batch_size, size_key="size", filter_param=filter_param)

    batch_count = 0
    cur_bucket = None
    counter = Counter()
    for idx, sample in enumerate(pipe.run(test_src(103))):
        if idx % batch_size == 0:
            # first sample of a batch
            batch_count += 1
            cur_bucket = sample['bucket']['idx']
            print(f"cur_bucket: {cur_bucket}, {sample}")
            counter.update({cur_bucket: 1})
        elif sample['bucket']['idx'] != cur_bucket:
            print(f"different bucket found in same batch: {cur_bucket}, {sample}")

    print(f"total batchs: {batch_count}. counter={counter}")

    pass
