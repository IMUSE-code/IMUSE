import torch
from torch import nn

from network.transformer_module import Decoder 

import math 

from tqdm.auto import tqdm

from einops import reduce

from inspect import isfunction

import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_feats,
        d_model,
        d_imu,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=self.d_feats + d_imu, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, padding_mask=None):
        # src: BS X T X D
        # noise_t: int 
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
    
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        d_feats,
        d_model,
        d_imu,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
    ):
        super().__init__()

        torch.random.manual_seed(0)

        self.denoise_fn = TransformerDiffusionModel(d_feats=d_feats, d_model=d_model, d_imu=d_imu, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, clip_denoised, padding_mask=None):
        x_all = torch.cat((x, x_cond), dim=-1)
        model_output = self.denoise_fn(x_all, t, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, clip_denoised=True, padding_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            clip_denoised=clip_denoised, padding_mask=padding_mask)
        noise = torch.randn_like(x).to(device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, cond_mask, padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond_mask, padding_mask=padding_mask)     

        return x # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window(self, shape, x_start, cond_mask):
        device = self.betas.device

        b = shape[0]
        assert b == 1
        
        x_all = torch.randn(shape, device=device)
        x_cond_all = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        x_blocks = []
        x_cond_blocks = []
        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = x_start.shape[1]
        stride = self.window // 2
        for t_idx in range(0, num_steps, stride):
            x = x_all[0, t_idx:t_idx+self.window]
            x_cond = x_cond_all[0, t_idx:t_idx+self.window]

            x_blocks.append(x) # T X D 
            x_cond.append(x_cond) # T X D 

        last_window_x = None 
        last_window_cond = None 
        if x_blocks[-1].shape[0] != x_blocks[0].shape[0]:
            last_window_x = x_blocks[-1][None] # 1 X T X D 
            last_window_cond = x_cond_blocks[-1][None] 

            x_blocks = torch.stack(x_blocks[:-1]) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks[:-1]) # K X T X D 
        else:
            x_blocks = torch.stack(x_blocks) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks) # K X T X D 
       
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x_blocks = self.p_sample(x_blocks, torch.full((b,), i, device=device, dtype=torch.long), x_cond_blocks)    

        if last_window_x is not None:
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                last_window_x = self.p_sample(last_window_x, torch.full((b,), i, device=device, dtype=torch.long), last_window_cond)    

        # Convert from K X T X D to a single sequence.
        seq_res = None  
        # for t_idx in range(0, num_steps, stride):
        num_windows = x_blocks.shape[0]
        for w_idx in range(num_windows):
            if w_idx == 0:
                seq_res = x_blocks[w_idx] # T X D 
            else:
                seq_res = torch.cat((seq_res, x_blocks[self.window-stride:]), dim=0)

        if last_window_x is not None:
            seq_res = torch.cat((seq_res, last_window_x[self.window-stride:]), dim=0)

        return seq_res # BS X T X D

    @torch.no_grad()
    def sample(self, x_start, cond_mask, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, cond_mask, padding_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window(self, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, cond_mask, t, noise=None, padding_mask=None):
        # x_start: BS X T X D
        # cond_mask: BS X T X D, missing regions are 1, head pose conditioned regions are 0.  
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
        masked_x_input = x 
        x_all = torch.cat((masked_x_input, cond_mask), dim=-1)
        model_out = self.denoise_fn(x_all, t, padding_mask)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # mask = torch.zeros(55).to(x.device)
        # mask[6:8] = 1
        # model_out *= mask
        # target *= mask
        # Predicting both head pose and other joints' pose. 
        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 
           
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

    def forward(self, x_start, cond_mask, padding_mask=None):
        # x_start: BS X T X D 
        # cond_mask: BS X T X D 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        # print("t:{0}".format(t))
        curr_loss = self.p_losses(x_start, cond_mask, t, padding_mask=padding_mask)

        return curr_loss
        