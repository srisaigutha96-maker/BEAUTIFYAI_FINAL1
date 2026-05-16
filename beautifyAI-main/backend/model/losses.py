import torch
import torch.nn as nn
import torchvision.models as models

# Try to use facenet_pytorch for Identity Loss if available
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    pass

class NonSaturatingGANLoss(nn.Module):
    """ Standard Non-Saturating Adversarial Loss """
    def __init__(self):
        super().__init__()
        
    def d_loss(self, real_preds, fake_preds):
        real_loss = nn.functional.softplus(-real_preds)
        fake_loss = nn.functional.softplus(fake_preds)
        return real_loss.mean() + fake_loss.mean()

    def g_loss(self, fake_preds):
        return nn.functional.softplus(-fake_preds).mean()

class PerceptualLoss(nn.Module):
    """
    Extracts features from an ImageNet-pretrained VGG19.
    Penalizes stylistic and feature differences as opposed to pure pixels.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval().to(device),
            vgg[4:9].eval().to(device),
            vgg[9:18].eval().to(device),
            vgg[18:27].eval().to(device),
            vgg[27:36].eval().to(device)
        ])
        for p in self.parameters():
            p.requires_grad = False
            
        self.criterion = nn.L1Loss()
        # VGG requires standard ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def _normalize(self, x):
        # We assume input is [-1, 1] coming from Generator
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, input, target):
        x = self._normalize(input)
        y = self._normalize(target)
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.criterion(x, y)
        return loss

class IdentityLoss(nn.Module):
    """
    Extracts high-level facial embeddings and calculates Cosine Distance.
    Crucial to prevent the GAN from 'redrawing' an entirely different person.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            # Requires `pip install facenet-pytorch`
            self.net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            for p in self.parameters():
                p.requires_grad = False
        except Exception as e:
            self.net = None
            print("ArcFace/InceptionResnet not loaded, Identity loss will be 0.")
            
    def forward(self, input, target):
        if self.net is None:
            return torch.tensor(0.0, device=input.device)
            
        # Network expects input around 160x160 centered face crop.
        # For simplicity, we interpolate the whole 512x512 image here.
        x = torch.nn.functional.interpolate(input, size=(160, 160), mode='bilinear')
        y = torch.nn.functional.interpolate(target, size=(160, 160), mode='bilinear')
        
        feat_x = self.net(x)
        feat_y = self.net(y)
        
        # 1 - Cosine Similarity = distance
        return 1.0 - torch.nn.functional.cosine_similarity(feat_x, feat_y, dim=1).mean()

class FiBGANLossAggregator(nn.Module):
    """ Combinator for the required Loss topologies specified """
    def __init__(self, device='cuda'):
        super().__init__()
        self.adv_loss = NonSaturatingGANLoss()
        self.perceptual_loss = PerceptualLoss(device)
        self.identity_loss = IdentityLoss(device)
        self.l1_loss = nn.L1Loss()
        
        # Hyperparameters controlling influence
        self.lambda_adv = 1.0
        self.lambda_perc = 1.0
        self.lambda_id = 0.5
        self.lambda_l1 = 10.0
        
    def get_d_loss(self, real_preds, fake_preds):
        loss_d = self.adv_loss.d_loss(real_preds, fake_preds)
        return loss_d

    def get_g_loss(self, fake_img, real_img, fake_preds):
        loss_g_adv = self.adv_loss.g_loss(fake_preds) * self.lambda_adv
        loss_perc = self.perceptual_loss(fake_img, real_img) * self.lambda_perc
        loss_id = self.identity_loss(fake_img, real_img) * self.lambda_id
        loss_l1 = self.l1_loss(fake_img, real_img) * self.lambda_l1
        
        total_g_loss = loss_g_adv + loss_perc + loss_id + loss_l1
        
        return total_g_loss, {
            'adv': loss_g_adv.item(),
            'perc': loss_perc.item(),
            'id': loss_id.item(),
            'l1': loss_l1.item()
        }