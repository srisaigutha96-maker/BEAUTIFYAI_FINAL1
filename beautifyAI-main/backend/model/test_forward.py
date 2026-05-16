import torch
from generator import Generator
from discriminator import MultiScaleDiscriminator
from losses import FiBGANLossAggregator

def test_fibgan_forward():
    print("Initializing Generator...")
    gen = Generator(style_dim=512)
    print("Initializing Discriminator...")
    disc = MultiScaleDiscriminator()
    print("Initializing Loss Aggregator (CPU mode)...")
    criterion = FiBGANLossAggregator(device='cpu')
    
    # Batch size 2, 3 Channels, 512x512 Res
    x_fake = torch.randn(2, 3, 512, 512)
    x_real = torch.randn(2, 3, 512, 512)
    
    # 1. Forward Pass
    print("Testing Generator Forward pass...")
    intensity = torch.tensor([[0.6], [0.9]])
    out = gen(x_fake, intensity)
    
    assert out.shape == (2, 3, 512, 512), f"Generator shape mismatch: {out.shape}"
    print("Generator forward successful!")
    
    # 2. Discriminator Forward Pass
    print("Testing Discriminator Forward pass...")
    d_preds = disc(out)
    assert len(d_preds) == 2, "Multiscale Discriminator should return two lists of predictions"
    print("Discriminator forward successful!")
    
    # 3. Loss Graph Forward
    print("Testing Losses...")
    total_g_loss = 0
    for pred in d_preds:
        loss, _ = criterion.get_g_loss(out, x_real, pred)
        total_g_loss += loss
        
    print(f"Total G Loss graph connection computed: {total_g_loss.item():.4f}")
    
    # 4. Dummy Backward
    print("Testing Backward Graph (Gradients)...")
    total_g_loss.backward()
    
    # Ensure gradients reached the start of the Generator (FPN)
    assert gen.fpn.enc1[0].weight.grad is not None, "Gradients did not flow backward through the Model!"
    print("Successfully mapped gradient flow backward through FiBGAN topologies!")

if __name__ == "__main__":
    test_fibgan_forward()


