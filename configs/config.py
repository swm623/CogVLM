###############global########################
feiyun = False

###############Train#########################
# wandb_project = '01txt2img'
# set run_id to continue from prev-run

# run_id="sdxl-17w-20231018_141642"
# load_ckpt = 'ckpts/sdxl-17w-20231018_141642'

enable_amp = True
dtype = "fp16"  # float32 fp16 bf16 fp8

num_gpus = 4
image_size = 1024

if image_size == 256:
    batch_size = 128
elif image_size == 512:
    batch_size = 8
elif image_size == 1024:  # multi aspect ratio
    # use 2 for 2.6B, or use 8 for 1.3B model
    batch_size = 2
    noise_offset = 0.05  # only add when image_size=1024

world_size = num_gpus
print_freq = 10  # don't print too often
test_freq = print_freq * 8
seed = 1337
compile = False
use_vae_encode_online = True
use_text_encode_online = True
num_max_save_models = 100
test_before_train = False

# loss discount of best loss while saving models
# eg. if current best loss is 0.1, models with loss less than 0.1/0.2 = 0.5 will be saved 
save_discount = 0.2

# set to true if model is differ than original
find_unused_parameters = False
###############Model EMA#####################
use_ema = False
ema_beta = 0.9995
ema_update_every = 20

###############Optimizer#####################
gradient_accumulation_steps = 16 if image_size == 1024 else 1
warm_steps = 80
total_steps = 10000
max_epochs = 10
learning_rate = 5e-6
min_lr = 1e-6  # learning_rate / 10 usually
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999  # make a bit bigger when batch size per iter is small


###############Diffusion#########################
objective = 'pred_noise'
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
num_workers = 1  # number of dataloader processes per GPU(process)

#### webdataset:
# suport both pre-embeded tar-files or raw-tar-files of laion-dataset
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
