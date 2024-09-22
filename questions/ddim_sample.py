import torch
from tqdm import trange
import numpy as np

def denoise_ddim(x, alpha_bar, alpha_bar_prev, pred_noise):
    """
    Denoises the input tensor `x` using the DDIM (Denoising Diffusion Implicit Models) algorithm.

    Args:
        x (torch.Tensor): The input tensor to be denoised.
        alpha_bar (torch.Tensor): The cumulative product of alpha values at the current timestep.
        alpha_bar_prev (torch.Tensor): The cumulative product of alpha values at the previous timestep.
        pred_noise (torch.Tensor): The predicted noise tensor.

    Returns:
        torch.Tensor: The denoised tensor.
    """
    # Predict the original image using the previous timestep's alpha_bar
    x0_pred = torch.sqrt(alpha_bar_prev) * ((x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar))
    
    # Calculate the direction term using the current timestep's alpha_bar and predicted noise
    direction_x_t = torch.sqrt(1 - alpha_bar_prev) * pred_noise 
    
    # Return the sum of the predicted image and the direction term
    return x0_pred + direction_x_t

@torch.no_grad()
def sample_ddim(neural_net, n_samples, timesteps, image_width, image_height, alpha_bars, alphas, betas, device, step_size=20):
    """
    Samples images using a DDIM (Denoising Diffusion Implicit Models) by iteratively denoising.

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
        step_size (int, optional): The step size for the timesteps. Defaults to 20.

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
    for timestep in trange(timesteps, 0, -step_size, desc='Sampling timesteps'):
        
        # Create timestep tensor normalized to the range [0, 1]
        t = torch.tensor([timestep / timesteps])[:, None, None, None].to(device)
        
        # Predict noise using the neural network
        predicted_noise = neural_net(samples, t)
        
        # Update samples by denoising and adding noise
        samples = denoise_ddim(samples, alpha_bars[timestep], alpha_bars[timestep - step_size], predicted_noise)

        # Save intermediate samples for visualization
        intermediate_samples.append(samples.detach().cpu().numpy())
            
    # Convert list of intermediate samples to a numpy array
    intermediate_samples = np.stack(intermediate_samples)
    
    return samples, intermediate_samples