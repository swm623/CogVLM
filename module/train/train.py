import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from diffusers import DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm

# local
from module.nn import VAE, TextEncoder
from module.utils import cprint, print, init_seeds
from module.dataloader import load_webdataset
import warnings

warnings.filterwarnings("ignore")


class TrainProcess:
    def __init__(self, rank, local_rank, opt):
        super().__init__()
        if rank == 0:
            cprint('#### Start main Process. pid=%d' % os.getpid(), 'red')
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = opt.world_size
        self.start_epoch = 0
        self.start_step = 0
        self.opt = opt
        self.current_lr = opt.learning_rate
        self.start_time = time.time()

        if opt.dtype == 'float32':
            self.dtype = torch.float32
        elif opt.dtype == 'fp16':
            self.dtype = torch.float16
        elif opt.dtype == 'bf16':
            self.dtype = torch.bfloat16

    def run(self):  # run one ervery process
        opt = self.opt
        rank = self.rank
        local_rank = self.local_rank
        cprint('#### Start run rank %d (local_rank=%d) Process. pid=%d' %
               (rank, local_rank, os.getpid()), 'red')
        torch.cuda.set_device(local_rank)
        # init seed
        init_seeds(opt.seed + rank)
        # init distribute env
        cprint('[rank-%d] init distribute env ...' % rank)
        # torch.distributed.init_process_group(backend='nccl')
        self.device = torch.device('cuda:{:d}'.format(self.local_rank))

        # set tensorboard
        if self.rank == 0:
            logpath = os.path.join(self.opt.model_save_dir, 'logs')
            self.sw = SummaryWriter(logpath)

        # set dataloader
        self.set_dataprovider(opt)
        # build model
        self.creat_model()

        self.test(self.start_step)



    def set_dataprovider(self, opt):
        cprint('[rank-%d] set dataloader ...' % self.rank, 'cyan')
        self.test_loader = load_webdataset(opt.test_data_dir,
                                               num_samples=opt.test_data_size,
                                               resolution=opt.resolution,
                                               num_workers=2,
                                               batch_size=opt.batch_size,
                                               world_size=opt.world_size,
                                               image_filter_param=opt.image_filter_param,
                                               random_crop=opt.random_crop,
                                               is_train=False,
                                               seed=opt.seed
                                               )

    def creat_model(self):
        cprint('[rank-%d] build model ...' % self.rank, 'cyan')
        opt = self.opt

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler")
        
        self.vae = VAE(opt.pretrained_model_name_or_path,
                           device=self.device, compile=opt.compile)
        
        self.text_encoder = TextEncoder(opt.pretrained_model_name_or_path,
                                            device=self.device)

        self.model = UNet2DConditionModel.from_pretrained(opt.pretrained_model_name_or_path,
                                                          subfolder="unet",
                                                          revision=opt.revision)  # .to(self.device)

        self.model = self.model.to(self.device)
        if opt.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=opt.find_unused_parameters)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=opt.learning_rate,
                                       weight_decay=opt.weight_decay, betas=[opt.beta1, opt.beta2])
        # self.optim = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=opt.learning_rate)

        if opt.compile:
            if self.rank == 0:
                print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        # self.loss_fn = MyLossFn(opt).to(device)
        self.load_model()

        # self.adjust_learning_rate_exponential(self.start_epoch, self.start_step)
        if self.rank == 0:
            num_parameters = 0
            for p in self.model.parameters():
                num_parameters += p.numel()
            cprint('Num parameters=%0.2fM' % (num_parameters / 1e6), color='red')


    def load_model(self):
        opt = self.opt
        filepath = None
        if hasattr(opt, 'load_ckpt') and isinstance(opt.load_ckpt, str):
            if os.path.isdir(opt.load_ckpt):
                filepath = self.get_lastest_ckpt(opt.load_ckpt)
            elif os.path.isfile(opt.load_ckpt):
                filepath = opt.load_ckpt
            else:
                cprint('checkpoint file not found! %s' % opt.load_ckpt, 'red', attrs=['blink'])

        if filepath is not None:
            cprint('load ckpt from %s' % filepath, on_color='on_red')
            checkpoint = torch.load(filepath, map_location='cpu')
            self.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            self.start_step = checkpoint['step'] if 'step' in checkpoint else 0
            # self.best = checkpoint['best'] if 'best' in checkpoint else 0
            ckpt = self.get_match_ckpt(self.model, checkpoint['model'])
            self.model.load_state_dict(ckpt)
            if 'optim' in checkpoint:
                self.optim.load_state_dict(checkpoint['optim'])
        return
    def calc_loss(self, model, batch):
        opt = self.opt

        # upload to gpu
        time_ids = batch['size_of_original_crop_target'].to(
            device=self.device, dtype=self.dtype)
        image = batch['image'].to(device=self.device, dtype=self.dtype)
        if image.shape[-1] == 3:
            image = torch.permute(image, [0, 3, 1, 2])  # NHWC->NCHW

        # do vae and text_encoder here
        x = self.vae(image)
        t = self.text_encoder(batch['text'])

        B, C, H, W = x.shape
        noise = torch.randn_like(x)
        if 'noise_offset' in opt and opt.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += opt.noise_offset * \
                torch.randn((B, C, 1, 1), device=x.device)

        unet_added_conditions = {"time_ids": time_ids,
                                 "text_embeds": t['pooled_prompt_embeds']}

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, opt.timesteps, (B,), device=x.device).long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(x, noise, timesteps)

        with autocast(enabled=True):
            model_pred = model(noisy_model_input, timesteps, t['prompt_embeds'],
                               added_cond_kwargs=unet_added_conditions).sample
            return F.mse_loss(model_pred.float(), noise, reduction="mean")


    def do_optimize(self, batch, step, scaler):
        self.model.train()
        opt = self.opt
        gradient_accumulation_steps = opt.gradient_accumulation_steps
        if opt.world_size > 1:
            self.model.require_backward_grad_sync = False if step % gradient_accumulation_steps != 0 else True

        loss = self.calc_loss(self.model, batch)

        mb_losses = {}
        mb_losses['total'] = loss
        # self.optim.zero_grad()
        scaler.scale(mb_losses['total'] / gradient_accumulation_steps).backward()
        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            scaler.step(self.optim)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optim.zero_grad(set_to_none=True)

        self.update_meter(mb_losses)
        return mb_losses

    @torch.no_grad()
    def test(self, step):
        self.model.eval()

        t0 = time.time()
        cnt, loss_sum = 0, 0
        for mb in self.test_loader:
            # mb = self.preprocess(mb)
            # # pred_batch return avg loss for batch
            # loss_sum += self.pred_batch(mb) * mb['latent'].shape[0]
            loss_sum += self.calc_loss(self.model, mb) * mb['image'].shape[0]
            cnt += mb['image'].shape[0]

        loss = loss_sum / cnt
        speed = cnt / (time.time() - t0)
        print(f'===num_test_images={cnt}  speed={speed:.2f}imgs/s loss={loss_sum / cnt:0.4f} ')
        self.sw.add_scalar("test/loss", loss, step)
        self.sw.add_scalar("test/speed", speed, step)
        return loss

 