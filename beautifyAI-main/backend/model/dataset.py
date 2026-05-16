import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class BeautifyDataset(Dataset):
    """
    Given an HQ dataset (like CelebA-HQ), produces pairs of:
    (Degraded_Image, Ground_Truth_Image).
    The degradation mimics low-resolution, blurry, or noisy web photos.
    """
    def __init__(self, root_dir, image_size=512):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        
        # Ground Truth Pipeline
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])
        
    def degrade_image(self, img):
        """ Artificially decimate the image to train the restorer """
        # 1. Blur
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5, 7])
            img = TF.gaussian_blur(img, kernel_size=kernel_size)
            
        # 2. Downsample & Upsample (Pixelate)
        scale = random.uniform(0.1, 0.4)
        w, h = img.size
        small_w, small_h = int(w*scale), int(h*scale)
        img = TF.resize(img, (small_h, small_w), interpolation=Image.BILINEAR)
        img = TF.resize(img, (h, w), interpolation=Image.BICUBIC)

        # ToTensor & Normalize
        img_t = TF.to_tensor(img)
        
        # 3. Add Gaussian Noise
        if random.random() < 0.5:
            noise = torch.randn_like(img_t) * 0.05
            img_t = img_t + noise
            img_t = torch.clamp(img_t, 0.0, 1.0)
            
        img_t = TF.normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img_t

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        hq_img = Image.open(img_path).convert('RGB')
        
        # Target
        real_img = self.gt_transform(hq_img)
        # Input
        lq_img = self.degrade_image(hq_img.resize((self.image_size, self.image_size)))
        
        return {'lq': lq_img, 'hq': real_img}
