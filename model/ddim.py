import torch
import torch.nn as nn
from model.ddpm import DDPM
from questions.ddim_sample import sample_ddim

class DDIM(DDPM):
    def __init__(self, unet_model, beta1, beta2, T, eta, device=torch.device("cpu")):
        super(DDIM, self).__init__(unet_model, beta1, beta2, T, device)
        self.eta = eta
        
    def sample(self, n_samples, image_width, image_height, device, step_size=20):
        samples, intermediates = sample_ddim(self.unet_model, n_samples, self.T, image_width, image_height, self.alphabar_t, self.alpha_t, self.beta_t, device, step_size)
        
        return samples, intermediates