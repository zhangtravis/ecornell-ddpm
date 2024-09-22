import torch
import torch.nn as nn
from questions.ddpm_sample import sample_ddpm

class DDPM(nn.Module):
    def __init__(self, unet_model, beta1, beta2, T, device=torch.device("cpu")):
        super(DDPM, self).__init__()
        
        self._generate_ddpm_schedules(beta1, beta2, T, device)
        
        self.unet_model = unet_model
        self.T = T
        self.device = device
    
    def forward(self, x):
        timestamps = torch.randint(1, self.T + 1, (x.shape[0],)).to(self.device)
        epsilon = torch.rand_like(x).to(self.device)
        
        sqrt_alphabar = torch.sqrt(self.alphabar_t)
        sqrt_one_minus_alphabar = torch.sqrt(1 - self.alphabar_t)
        
#         import ipdb; ipdb.set_trace()
        
        x_t = sqrt_alphabar[timestamps, None, None, None] * x + sqrt_one_minus_alphabar[timestamps, None, None, None] * epsilon
        
        return self.unet_model(x_t, timestamps / self.T, self.T), epsilon
        
    def sample(self, n_samples, image_width, image_height, device, save_rate=20):
        samples, intermediates = sample_ddpm(self.unet_model, n_samples, self.T, image_width, image_height, self.alphabar_t, self.alpha_t, self.beta_t, device, save_rate)
        
        return samples, intermediates
    
    def _generate_ddpm_schedules(self, beta1, beta2, T, device):
        
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        
#         import ipdb; ipdb.set_trace()
#         self.beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32).to(device) / T + beta1
        self.beta_t = torch.linspace(beta1, beta2, T + 1, dtype=torch.float32).to(device)
#         self.sqrt_beta_t = torch.sqrt(beta_t)
        self.alpha_t = 1 - self.beta_t
        log_alpha_t = torch.log(self.alpha_t)
        self.alphabar_t = torch.cumprod(self.alpha_t, dim=0)
#         self.alphabar_t[0] = 1

#         self.sqrt_alphabar = torch.sqrt(self.alphabar_t)
#         self.oneover_sqrt_alpha = 1 / torch.sqrt(self.alpha_t)

#         self.sqrt_one_minus_alphabar = torch.sqrt(1 - self.alphabar_t)
#         self.one_minus_alpha_over_sqrt_one_minus_alphabar = (1 - alpha_t) / self.sqrt_one_minus_alphabar