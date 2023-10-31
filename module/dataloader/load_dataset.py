# coding=utf-8

import logging
import functools
import os
import webdataset as wds
import re
from collections import Counter
from .image_process import ImageProcess
from .bucket_batcher import AspectRatioBucketBatcher


logger = logging.getLogger(__name__)

_ext_fields_text = ('latent', 'size_of_original_crop_target',
                    'prompt_embeds', 'pooled_prompt_embeds')


def _normalize_caption(caption):
    # some caption is words seperated by specail char such as `_`, `-`,
    # and some caption is words starts with upcase char and there is not seprated char.
    caption = re.sub(r'%20', ' ', caption)
    caption = re.sub(r'"+', '"', caption)
    caption = re.sub(r"'+", "'", caption)
    if len(caption.split()) < 3:
        counter = Counter()
        for c in caption:
            if c.isalnum() or c.isspace():
                continue
            counter.update(c)

        for sep, cnt in counter.most_common(1):
            if cnt > 1 and sep not in "\"'":
                if sep not in '-,_.+/':
                    print(f"[{sep}:{cnt}]: {caption}")
                caption = caption.replace(sep, ' ')

    # split by Upcase
    prev = None
    r = ""
    for c in caption:
        if prev is None:
            prev = c
            continue
        r = r + prev
        if (prev.islower() and c.isupper()) or (c.isalpha() and prev.isdigit() and c != 'x'):
            r = r + " "
        prev = c
    r = r + prev

    return re.sub(r'\s+', ' ', r).strip()


def _wds_map_fun(sample: dict, image_proc_fun, trace_id=False):
    # support decode for origin laion dataset or pre-embeded laoin dataset

    # try retrieve pre-embed-info from npz or sample itself
    s = sample['npz'] if 'npz' in sample else sample
    r = {}
    for name in _ext_fields_text:
        if name in s:
            r[name] = s[name]

    # check if pre-embed-info is ready
    if 'latent' not in r:
        # should do vae online
        image, size_of_original_crop_target = image_proc_fun(
            sample['image'], key=sample['__key__'], meta=sample.get('json'), bucket=sample.get('bucket'))
        r.update(
            {"image": image, "size_of_original_crop_target": size_of_original_crop_target})

    if 'prompt_embeds' not in r:
        # should do text_encoding online, retrieve from text or json
        if 'txt' in sample:
            r['text'] = _normalize_caption(sample['txt'])
        elif 'json' in sample:
            r['text'] = sample['json']['caption']

    if trace_id:
        logger.info(os.getpid(), sample["__key__"])
    return r


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def filter_samples_by_name(sample: dict):
    """
        called when sample is generated and before image is decoded
    """
    if "txt" not in sample and 'json' not in sample:
        return False
    if 'jpg' not in sample and 'png' not in sample and "jpeg" not in sample:
        return False
    return True


# def filter_samples_by_content(sample: dict, drop_small_image_size=None, drop_image_ratio=None, **kwargs):
#     """
#         called after image is decoded
#         drop small-image and image with too high or too wide here.  
#         sample['image'] is type of numpy.ndarray        
#     """
#     if drop_small_image_size is not None:
#         if sample['image'].shape[0] < drop_small_image_size and sample['image'].shape[1] < drop_small_image_size:
#             logger.warning(
#                 f"dropping image by size: key={sample['__key__']}, size={sample['image'].shape}")
#             return False

#     if drop_image_ratio is not None:
#         ratio = sample['image'].shape[0] / sample['image'].shape[1]
#         if ratio > drop_image_ratio or ratio < 1.0 / drop_image_ratio:
#             logger.warning(
#                 f"dropping image by ratio: key={sample['__key__']}, size={sample['image'].shape}, ratio={ratio}")
#             return False
#     return True


def load_webdataset(datafiles, resolution, num_samples, num_workers, seed, batch_size,
                    world_size, is_train=True,
                    random_crop=False, dump_image=False, **kwargs):
    """
        suport pre-embeded-tar-files or raw-tar-files of laion-dataset
        datafiles: 
            webdataset supported format (file-pattern). e.g.: //path/to/file{0000..0005}.tar
            or data dir. e.g.: //path/to/data_dir

        return: dataloader
    """

    if isinstance(datafiles, str) and os.path.isdir(datafiles):
        filelist = []
        for dir, _, files in os.walk(datafiles):
            filelist.extend([os.path.join(dir, fn) for fn in files if fn.endswith('.tar')])
        datafiles = filelist

    image_proc = ImageProcess(resolution=resolution, random_crop=random_crop, dump_image=dump_image)
    preprocessor_fn = functools.partial(_wds_map_fun, image_proc_fun=image_proc)

    if 'image_filter_param' in kwargs:
        image_filter_param = kwargs.pop('image_filter_param')
    else:
        image_filter_param = {}

    # filter_samples_by_content_fn = functools.partial(filter_samples_by_content,
    #                                                  **image_filter_param)

    _num_workers = max(1, num_workers)

    pipeline = [
        wds.SimpleShardList(datafiles)
    ]
    if is_train:
        pipeline.extend([
            wds.shardlists.split_by_node,
            wds.shardlists.split_by_worker,
            # shuffle by shard
            wds.detshuffle(bufsize=1000, initial=100, seed=seed,
                           # epoch=shared_epoch,
                           ),
            # wds.tarfile_to_samples(handler=log_and_continue),
            wds.tarfile_to_samples(),
            # shuffle by sample
            wds.shuffle(bufsize=2000, initial=1000)
        ])
    else:
        # for test, running on rank=0
        if _num_workers > len(datafiles):
            num_workers = _num_workers = len(datafiles)
        pipeline.extend([
            wds.shardlists.split_by_worker,
            wds.tarfile_to_samples(),
        ])

    pipeline.extend([
        wds.select(filter_samples_by_name),
        wds.decode("rgb8", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp"),
        # wds.select(filter_samples_by_content_fn),
        AspectRatioBucketBatcher(
            batch_size=batch_size,  
            image_size_base=resolution,
            # 1024 -> 64 
            image_size_step=resolution//16,
            image_key='image', 
            filter_param=image_filter_param
        ),
        wds.map(preprocessor_fn),
        # wds.to_tuple("image", "size_of_original_crop_target", "text"),
        # wds.batched(batch_size, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        global_batch_size = batch_size * world_size
    else:
        global_batch_size = batch_size

    num_batches = num_samples // global_batch_size
    num_worker_batches = num_batches // _num_workers  # per dataloader worker
    num_batches = num_worker_batches * _num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches * batch_size)

    dl = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )

    dl.num_batches = num_batches
    dl.num_samples = num_samples

    return dl
