import argparse
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import matplotlib.pyplot as plt
import os

from model.ddpm import DDPM
from model.ddim import DDIM
from model.unet_resnet import UNet

from dataset import CelebADataset

from torch.optim.lr_scheduler import CosineAnnealingLR

def print_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params:,}')
    
def save_grid(intermediate_samples, save_dir):
    n_images, n_samples, c, h, w = intermediate_samples.shape
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(n_images):
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axs = axs.flatten()
        
        for j in range(n_samples):
            img = np.transpose(intermediate_samples[i, j], (1, 2, 0))  # Convert to HWC format
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            axs[j].imshow(img)
            axs[j].axis('off')
        
        for j in range(n_samples, len(axs)):
            axs[j].axis('off')
        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(save_dir, f'grid_{i:03d}.png'))
        plt.close()

parser = argparse.ArgumentParser(description='Train Diffusion Model')

parser.add_argument('--data_path', type=str, default='eurecom-ds/celeba', help='path to dataset')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--resume', type=str, help='Resume from this path')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size of dataset')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--model', type=str, default='ddpm', help='ddpm or ddim', choices=['ddpm', 'ddim'])
parser.add_argument('--gpu_id', type=int, default=0, choices=[0,1,2,3], help='which gpu to train on')
parser.add_argument('--dropout_val', type=float, default=0.3, help="Dropout value")

# Parse the arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if __name__ == '__main__':
    wandb.init(project="ecornell-ddpm")
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epoch
    model_type = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Creating UNet Model...")
    unet_model = UNet(in_channels=3, out_channels=3, n_feat=32, dropout_val=args.dropout_val, n_heads=2)
    unet_model.to(device)
    print(unet_model)
    
    print(f"Creating {model_type}...")
    model = None
    if model_type == "ddpm":
        model = DDPM(unet_model=unet_model, beta1=1e-4, beta2=0.02, T=1000, device=device)
    else:
        model = DDIM(unet_model=unet_model, beta1=1e-4, beta2=0.02, T=1000, eta=0.5, device=device)
        
    print_total_params(model)
    
    model.to(device)
        
    print("Creating and loading dataset...")    
    train_transform = transforms.Compose(  # resize to 512 x 512, convert to tensor, normalize
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
#             transforms.Lambda(lambda x: 2 * x / 255.0 - 1)
#             transforms.Normalize((0.5,), (0.5,))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    ds = load_dataset(args.data_path)
    train_dataset = CelebADataset(ds['train'], transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=len(train_dataloader) * epochs, last_epoch=-1, eta_min=1e-9)

    criterion = nn.MSELoss()
    
    wandb.watch(model, log='all')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}: ")
        model.train()
        
        loss_ema = None
        pbar = tqdm(train_dataloader)
        for images in pbar:
            images = images.to(device)
            
            optim.zero_grad()
#             import ipdb; ipdb.set_trace()
            pred, gt = model(images)
            loss = criterion(pred, gt)
            
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            
        wandb.log({"Training Loss": loss_ema})
        
        print("Evaluating model...")
        model.eval()
        with torch.no_grad():
            xh, intermediate_samples = model.sample(n_samples=32, image_width=32, image_height=32, device=device)
            xh = ((xh + 1) / 2) * 255
            xh = xh.clamp(0, 255).to(torch.int32)
            xset = torch.cat([xh, images[:32]], dim=0)
            grid = make_grid(xset, nrow=8)
            save_image(grid, f"./generations/{model_type}/sample_celeba_32_{epoch:03d}.png")
            wandb.log({"sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in xh]})
#             save_grid(intermediate_samples, f"./generations/{model_type}/intermediate_samples_celeba{epoch:03d}")

            # save model
            state = {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "epoch": epoch + 1,
                "model_type": model_type
            }
            
            torch.save(state, f"./{model_type}_celeba_32.pth")
            
    