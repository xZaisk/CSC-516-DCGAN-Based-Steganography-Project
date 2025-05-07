import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ----------- CONFIGURE THESE -----------
image_path = "output/stego_result.png"  # path to your generated image
decoder_path = "output/netDec.pth"      # path to trained decoder
nz = 100                                # latent vector size (must match training)
image_size = 64                         # image size (must match training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- DEFINE DECODER -----------
class Decoder(nn.Module):
    def __init__(self, nz=100):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, nz),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ----------- MESSAGE DECODER -----------
def noise_to_message(decoded_z, threshold=0.0):
    bits = ['1' if val > threshold else '0' for val in decoded_z.squeeze().tolist()]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars).strip('\x00'), bits

# ----------- LOAD & DECODE -----------
# Load decoder
netDec = Decoder(nz).to(device)
netDec.load_state_dict(torch.load(decoder_path, map_location=device))
netDec.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Decode
with torch.no_grad():
    recovered_z = netDec(image_tensor).detach().cpu()

decoded_msg, _ = noise_to_message(recovered_z)

print(f"\nDecoded message from image:\n{decoded_msg}")
