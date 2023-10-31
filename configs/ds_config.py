###############global########################
feiyun = False

###############Train#########################
dtype = "fp16"  # float32 fp16 bf16 fp8

image_size = 1024

# batch size per gpu = 2 * 32 = 64
# 160000/64 = 2500 batch for one gpu, and about 300 batch on one node
# 
print_freq = 10
test_freq = 100
save_freq = 100

seed = 1337
compile = False

# loss discount of best loss while saving models
# eg. if current best loss is 0.1, models with loss less than 0.1/0.2 = 0.5 will be saved 
save_discount = 0.2

# output_path: for ckpts and ds_logs
output_path = "output"
ckpt_path = output_path + "/ckpts"

###############Model EMA#####################
use_ema = False
ema_beta = 0.9995
ema_update_every = 20

###############Optimizer#####################
max_epochs = 6

###############Diffusion#########################
objective = 'pred_noise'
noise_offset = 0.05  # only add when image_size=1024
timesteps = 1000
sampling_timesteps = timesteps
ddim_sampling_eta = 1
min_snr_loss_weight = True
min_snr_gamma = 10
if 1:
    beta_schedule = 'sigmoid'
    schedule_fn_kwargs = {'start': -3, 'end': 3, 'tau': 0.9}
else:
    beta_schedule = 'cosine'
    schedule_fn_kwargs = {'start': 0, 'end': 0.5, 'tau': 1}

###############Net Structure#########################
if feiyun:
    pretrained_model_name_or_path = '/pfs/sshare/app/zhangsan/models/stable-diffusion-xl-base-1.0/'
    # pretrained_model_name_or_path = '/pfs/sshare/app/zhangsan/models/stable-diffusion-xl-base-1.0-1.3B/'
else:
    pretrained_model_name_or_path = '/ML-A100/sshare-app/zhangsan/models/stable-diffusion-xl-base-1.0/'  # 2.6B
    # pretrained_model_name_or_path = '/ML-A100/sshare-app/wangtao/models/stable-diffusion-xl-base-1.0/'  # 1.3B [0,2,4]
revision = None

###############DataLoader#####################
num_workers = 1  # number of dataloader processes per GPU(process), need > 0 for prefetch

#### webdataset:
# dir maybe a directory or urls that webdataset supported 
if feiyun:
    # train_data_dir = '/nfs/sshare/app/data/datasets/laion-high-resolution-output/{00000..00099}.tar'
    # test_data_dir = '/nfs/sshare/app/data/datasets/laion-high-resolution-output/{10000..10099}.tar'
    train_data_dir = '/pfs/sshare/app/dataset/laion-high-aesthetics_6/{000..089}.tar'
    test_data_dir = '/pfs/sshare/app/dataset/laion-high-aesthetics_6/{090..101}.tar'
else:
    # laion-high-resolution, total 10137 tarfiles, about 4000 sample per tarfile 
    # train_data_dir = '/ML-A100/sshare-app/zhangsan/laion-high-resolution-output/{00000..00099}.tar'
    # test_data_dir = '/ML-A100/sshare-app/zhangsan/laion-high-resolution-output/{10000..10099}.tar'
    # 
    # laion-high-aesthetics_6 is subset of laion-high-resolution-output with aesthetics > 6. about 17.4w
    train_data_dir = '/ML-A100/sshare-app/yanjinbing/laion-high-aesthetics_6/{000..089}.tar'
    test_data_dir = '/ML-A100/sshare-app/yanjinbing/laion-high-aesthetics_6/{090..101}.tar'

test_data_size = 128

# provide sample num here since webdataset do not provide total num
train_data_size = 160000
# drop images with short edge size less than `drop_small_image_size`
image_filter_param = {
    "drop_small_image_size": 512,
    "drop_image_aspect_ratio": 4.0,
}

###############ImageProcessing#####################
resolution = image_size
random_crop = True


###############DeepSeepConfig#####################
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    # print_freq for deepspeed
    "steps_per_print": 100,
    "dump_state": False,

    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-6,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": 5e-6,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    },
    "zero_optimization": {
        "stage": 2
    },
    "tensorboard": {
        "enabled": True,
        "output_path": output_path + "/ds_logs",
    },
    "csv_monitor": {
        "enabled": True,
        "output_path": output_path + "/ds_logs",
    },
    "wandb": {
        "enabled": True,
        # "group": "",
        # "team": "",
        "project": "01txt2img"
    },
    "autotuning": {

    }
}
