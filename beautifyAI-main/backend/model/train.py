import os
import torch
from torch.utils.data import DataLoader
from dataset import BeautifyDataset
from generator import Generator
from discriminator import MultiScaleDiscriminator
from losses import FiBGANLossAggregator
from torch.optim import Adam

# ================================
# checkpoint for later pause
# ================================

def load_latest_checkpoint(checkpoints_dir, gen, disc, opt_g, opt_d):
    import glob
    import re

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "fibgan_epoch_*.pth"))
    if not checkpoints:
        return 0  # start from epoch 0

    latest = max(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    print(f"Loading checkpoint: {latest}")

    checkpoint = torch.load(latest)
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    opt_g.load_state_dict(checkpoint['opt_g'])
    opt_d.load_state_dict(checkpoint['opt_d'])

    start_epoch = int(re.findall(r'\d+', latest)[-1]) + 1
    return start_epoch



def train_fibgan(data_dir, epochs=100, batch_size=4, lr=2e-4, checkpoints_dir='checkpoints'):
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training FiBGAN on {device}...")

    # Load dataset
    dataset = BeautifyDataset(root_dir=data_dir, image_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize Networks
    gen = Generator(style_dim=512).to(device)
    disc = MultiScaleDiscriminator(in_channels=3).to(device)
    
    # Initialize Aggregated Losses
    criterion = FiBGANLossAggregator(device=device)

    # Optimizers
    opt_g = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            lq_imgs = batch['lq'].to(device)
            hq_imgs = batch['hq'].to(device)
            
            # Simulated Random User Beautification Intensity choice during training [0.3, 0.6, 0.9, 1.0]
            intensity = torch.empty(lq_imgs.size(0), 1).uniform_(0.3, 1.0).to(device)
            
            # =======================
            # 1. Train Discriminator
            # =======================
            opt_d.zero_grad()
            
            # Real
            real_preds = disc(hq_imgs)
            
            # Fake
            with torch.no_grad():
                fake_imgs = gen(lq_imgs, intensity)
            fake_preds = disc(fake_imgs.detach())
            
            # The MultiScale discriminator returns a list of outputs [out1, out2]
            # We average the loss across both scales
            loss_d = 0
            for r_pred, f_pred in zip(real_preds, fake_preds):
                loss_d += criterion.get_d_loss(r_pred, f_pred)
            
            loss_d.backward()
            opt_d.step()
            
            # =======================
            # 2. Train Generator
            # =======================
            opt_g.zero_grad()
            
            # Generate new fakes for backward pass
            fake_imgs = gen(lq_imgs, intensity)
            fake_preds = disc(fake_imgs)
            
            # We aggregate fake predictions from both scales
            loss_g_total = 0
            for f_pred in fake_preds:
                loss, loss_dict = criterion.get_g_loss(fake_imgs, hq_imgs, f_pred)
                loss_g_total += loss
                
            loss_g_total.backward()
            opt_g.step()

            # Logging
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(loader)}] [D loss: {loss_d.item():.4f}] [G adv: {loss_dict['adv']:.4f}, perc: {loss_dict['perc']:.4f}, id: {loss_dict['id']:.4f}, l1: {loss_dict['l1']:.4f}]")

        # Save checkpoint end of epoch
        torch.save({
            'generator': gen.state_dict(),
            'discriminator': disc.state_dict(),
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict(),
        }, os.path.join(checkpoints_dir, f"fibgan_epoch_{epoch}.pth"))
        print(f"Saved Checkpoint for Epoch {epoch}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to high-quality CelebA images")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    train_fibgan(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)

