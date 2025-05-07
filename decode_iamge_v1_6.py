import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ------------------ CONFIG ------------------
# Make sure these paths and values match your training setup
image_path = "output/stego_result.png"   # path to the generated image
decoder_path = "output/netDec.pth"       # path to the trained decoder weights
nz = 100                                 # size of the latent vector
image_size = 64                          # image resolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ DECODER DEFINITION ------------------
class Decoder(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, nz),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ------------------ UTILITY FUNCTION ------------------
def noise_to_message(decoded_z, threshold=0.0):
    # converts the decoded latent vector to a string, each float is thresholded to a bit and thjen the bits are grouped into characters.
    bits = ['1' if val > threshold else '0' for val in decoded_z.squeeze().tolist()]
    chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars).strip('\x00'), bits

# ------------------ LOAD MODEL ------------------
netDec = Decoder(nz).to(device)
netDec.load_state_dict(torch.load(decoder_path, map_location=device))
netDec.eval()

# ------------------ LOAD IMAGE ------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# ------------------ DECODE MESSAGE ------------------
with torch.no_grad():
    recovered_z = netDec(image_tensor).cpu()

decoded_msg, _ = noise_to_message(recovered_z)

# ------------------ OUTPUT ------------------
print("\n🧠 Decoded message from image:")
print(decoded_msg)
