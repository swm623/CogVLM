import os
import argparse
import torch
from pprint import pprint
from module.utils import AttrDict, Util, cprint, get_opt_from_python_config
from module.train import TrainProcess

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='tmp', help='tag')
    parser.add_argument('--config_file', type=str, default='configs/config.py', help='')
    opt = parser.parse_args()
    opt = AttrDict(vars(opt))
    return opt

def main(rank, local_rank):
    if rank == 0:
        cprint('=> torch version : {}'.format(torch.__version__), 'blue')
        cprint('Initializing Denoising Diffusion Training Process..', 'red')

    opt2 = parse_arguments()
    opt = get_opt_from_python_config(opt2.config_file)
    opt.update(opt2)
 
    if 'run_id' not in opt:
        opt.run_id = Util.gen_run_id(opt.tag)
    opt.model_save_dir = 'ckpts/' + opt.run_id
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1)) 

    cprint(f"WORLD_SIZE: {opt.world_size}, RANK: {rank}")    

    if rank == 0:
        tensorboard_logpath = os.path.join(opt.model_save_dir, 'logs')
        os.system('rm -r %s/*.*' % tensorboard_logpath)
        pprint(vars(opt))

    p = TrainProcess(rank, local_rank, opt)
    p.run()


if __name__ == '__main__':
    rank = int(os.environ.get('RANK', 0))
    local_rank =int(os.environ.get('LOCAL_RANK', 0))
    main(rank, local_rank)
