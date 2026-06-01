import sys
import os

# Ensure the model directory is on the path so 'generator' can be imported
# regardless of what the current working directory is.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

import argparse


# ── Pillow-based fallback enhancement (no deep-learning required) ─────────
def _pillow_enhance(input_path, output_path, intensity):
    """Apply simple brightness/contrast/sharpness enhancement with Pillow."""
    from PIL import Image, ImageEnhance
    img = Image.open(input_path).convert("RGB")
    factor = 1.0 + intensity * 0.5          # e.g. 0.5 intensity → 1.25x
    img = ImageEnhance.Brightness(img).enhance(1.0 + intensity * 0.15)
    img = ImageEnhance.Contrast(img).enhance(factor)
    img = ImageEnhance.Sharpness(img).enhance(factor)
    img = ImageEnhance.Color(img).enhance(1.0 + intensity * 0.2)
    img.save(output_path, quality=95)
    print(f"[Fallback] Pillow enhancement saved to {output_path}")


# ── Deep-learning inference ──────────────────────────────────────────────
def _dl_enhance(input_path, output_path, weights_path, intensity, device):
    import torch
    from torchvision import transforms
    from PIL import Image
    from generator import Generator

    def load_generator(checkpoint_path, dev):
        gen = Generator(style_dim=512).to(dev)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=dev)
        if 'generator' in state_dict:
            gen.load_state_dict(state_dict['generator'])
        else:
            gen.load_state_dict(state_dict)
        gen.eval()
        return gen

    def process_image(img_path, dev):
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        x = transform(img).unsqueeze(0).to(dev)
        return x, orig_size

    def save_image(tensor, out_path, orig_size):
        tensor = (tensor.squeeze(0) + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor.cpu())
        img = img.resize(orig_size, Image.LANCZOS)
        img.save(out_path)

    # Fast Face Check using OpenCV Haar Cascades
    try:
        import cv2
        import os
        import sys
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        image = cv2.imread(input_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) == 0:
                print("NO_FACE_DETECTED", file=sys.stderr)
                sys.exit(2)
    except Exception as e:
        print(f"Face check failed: {e}", file=sys.stderr)

    print(f"Loading Generator on {device}...")
    gen = load_generator(weights_path, device)
    x, orig_size = process_image(input_path, device)
    print(f"Processing with intensity: {intensity}")
    with torch.no_grad():
        raw_out = gen(x, intensity=intensity)
        # Preserve identity based on any intensity level by blending with the original input
        # At max intensity (1.0), we retain 50% of original pixels to guarantee identity preservation
        alpha = float(intensity) * 0.5
        out = x * (1.0 - alpha) + raw_out * alpha
    save_image(out, output_path, orig_size)
    print(f"Saved beautified image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("--intensity", "-intensity", type=float, default=1.0)
    args = parser.parse_args()

    # Resolve weights path relative to model directory if not absolute
    weights_abs = args.weights if os.path.isabs(args.weights) else os.path.join(_MODEL_DIR, args.weights)

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dl_enhance(args.input, args.output, weights_abs, args.intensity, device)
    except Exception as e:
        print(f"[Warning] Deep-learning inference failed: {e}")
        print("[Fallback] Using Pillow-based enhancement instead...")
        try:
            _pillow_enhance(args.input, args.output, args.intensity)
        except Exception as fe:
            print(f"[Error] Pillow fallback also failed: {fe}")
            sys.exit(1)
