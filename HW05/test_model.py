import argparse
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as F

from model import MLP


parser = argparse.ArgumentParser()
parser.add_argument("img_path", type=str)
args = parser.parse_args()

img_float32 = read_image(args.img_path)[:3, :, :].float() / 255.0
img_grayscale = F.rgb_to_grayscale(img_float32, num_output_channels=1)
inputs = 1 - img_grayscale

print(f"Image shape: {img_grayscale.shape}")
for width in inputs.squeeze(0):
    print(" ".join(f"{pixel.item():.1f}" for pixel in width))

out_path = "model.pt"
model = MLP()
model.load_state_dict(torch.load(out_path))

with torch.no_grad():
    outputs = model(inputs)
    _, pred = torch.max(outputs, dim=1)
print(f"Model's prediction is {pred[0]}")
