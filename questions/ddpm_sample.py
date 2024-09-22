import torch
from tqdm import trange
import numpy as np

def denoise_and_add_noise(x, predicted_noise, beta_t, alpha_t, alpha_bar_t, device, z=None):
    """
    Denoises the input tensor `x` using the predicted noise and specified parameters, 
    and then adds noise to it based on the same parameters.

    Args:
        x (torch.Tensor): The input tensor to be denoised.
        predicted_noise (torch.Tensor): The predicted noise tensor.
        beta_t (torch.Tensor): The beta parameter at time step t.
        alpha_t (torch.Tensor): The alpha parameter at time step t.
        alpha_bar_t (torch.Tensor): The cumulative product of alpha parameters up to time step t.
        z (torch.Tensor, optional): A tensor representing noise to be added. If None, 
                                    standard normal noise is generated. Defaults to None.

    Returns:
        torch.Tensor: The tensor resulting from denoising `x` and adding noise.
    """
    
    # If no noise tensor `z` is provided, generate a tensor of standard normal noise with the same shape as `x`
    if z is None:
        z = torch.randn_like(x, device=device)
    
    # Compute the noise to be added using the square root of beta_t and the noise tensor `z`
    # HINT: You may find torch.sqrt() useful
    noise = torch.sqrt(beta_t) * z
    
    # Calculate the mean for the denoised data
    mean = torch.clamp(1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * predicted_noise), min=-1, max=1)
    
    # Return the denoised data with added noise
    return mean + noise

@torch.no_grad()
def sample_ddpm(neural_net, n_samples, timesteps, image_width, image_height, alpha_bars, alphas, betas, device, save_rate=20):
    """
    Samples images using a diffusion model by iteratively denoising and adding noise.

    Args:
        neural_net (nn.Module): The neural network model used to predict noise.
        n_samples (int): The number of samples to generate.
        timesteps (int): The total number of timesteps for the diffusion process.
        image_width (int): The width of the generated images.
        image_height (int): The height of the generated images.
        alpha_bars (list of torch.Tensor): List of cumulative alpha values for each timestep.
        alphas (list of torch.Tensor): List of alpha values for each timestep.
        betas (list of torch.Tensor): List of beta values for each timestep.
        device (torch.device): The device to run the computations on (CPU or GPU).
        save_rate (int, optional): The frequency of saving intermediate samples for visualization. Defaults to 20.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The final generated samples.
            - np.ndarray: An array of intermediate samples for visualization.
    """
    
   # Sample initial noise tensor with standard normal distribution; x_T ~ N(0, 1)
    samples = torch.randn(n_samples, 3, image_width, image_height).to(device)
    
    # List to store intermediate samples for visualization
    intermediate_samples = []
    
    # Iterate through timesteps in reverse order
    for timestep in trange(timesteps, 0, -1, desc='Sampling timesteps'):
        
        # Create timestep tensor normalized to the range [0, 1]
        t = torch.tensor([timestep / timesteps]).to(device)
        
        # Generate or use zero noise depending on the current timestep
        z = torch.randn_like(samples) if timestep > 1 else 0
        
#         import ipdb; ipdb.set_trace()
        # Predict noise using the neural network
        predicted_noise = neural_net(samples, t, timesteps)
        
        # Update samples by denoising and adding noise
        samples = denoise_and_add_noise(samples, predicted_noise, betas[timestep], alphas[timestep], alpha_bars[timestep], device, z)
        # Save intermediate samples at specified intervals
        if timestep % save_rate == 0 or timestep == timesteps or timestep < 8:
            intermediate_samples.append(samples.detach().cpu().numpy())
            
    # Convert list of intermediate samples to a numpy array
    intermediate_samples = np.stack(intermediate_samples)
    
    return samples, intermediate_samples
    