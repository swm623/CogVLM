import os
import shutil
import random
import torch
import termcolor
import builtins
import subprocess


def cprint(*args, **kwargs):
    kwargs['flush'] = True
    # builtins.print(*args, **kwargs)
    # if len(args) > 1:
    #     args = [str(a) for a in args]
    #     args = (''.join(args),)
    termcolor.cprint(*args, **kwargs)


def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)


def get_gpu_id(num_gpus=8):
    try:
        import wtutils
        return wtutils.get_gpu_id(num_gpus)
    except Exception as e:
        return ','.join([str(i) for i in range(num_gpus)])


def getFolderList(directory, join=False):
    folders = os.listdir(directory)

    folders_match = []
    for fd in folders:
        fd_join = os.path.join(directory, fd)
        if os.path.isdir(fd_join):
            if join:
                folders_match.append(fd_join)
            else:
                folders_match.append(fd)
    return folders_match


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_module_from_file(name, path):
    import importlib
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_opt_from_python_config(config_file):
    X = load_module_from_file('', config_file)

    config_keys = [k for k in dir(X) if not k.startswith('_')]
    config = {k: getattr(X, k) for k in config_keys}
    opt = AttrDict(config)
    return opt


# def build_env(config, config_name, ckpt_path):
#     t_path = os.path.join(ckpt_path, config_name)
#     if config != t_path:
#         os.makedirs(ckpt_path, exist_ok=True)
#         shutil.copyfile(config, os.path.join(ckpt_path, config_name))


def init_seeds(seed=0):
    # seed = 0
    # sets the seed for generating random numbers.
    torch.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    if seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_port():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    used_ports = subprocess.getoutput(pscmd).split()
    port = str(random.randint(15000, 20000))
    if port not in used_ports:
        return port
    else:
        get_port()
