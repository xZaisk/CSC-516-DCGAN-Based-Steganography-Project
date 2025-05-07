import torch
import torch.nn as nn
from torchvision import utils
import os

# ---------- CONFIGURATION ----------
save_dir = "output"
generator_path = os.path.join(save_dir, "netG.pth")
nz = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)

# ---------- GENERATOR DEFINITION ----------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ---------- MESSAGE ENCODING ----------
def message_to_noise(message, nz=100):
    bits = ''.join(format(ord(c), '08b') for c in message)
    bits = bits[:nz] + '0' * (nz - len(bits))
    z = torch.tensor([1.0 if b == '1' else -1.0 for b in bits], dtype=torch.float32)
    return z.view(1, nz, 1, 1).to(device)

# ---------- LOAD GENERATOR ----------
netG = Generator().to(device)
netG.load_state_dict(torch.load(generator_path, map_location=device))
netG.eval()

# ---------- INPUT MESSAGE ----------
message = input("Enter a message to embed: ")
z = message_to_noise(message, nz)

# ---------- GENERATE IMAGE ----------
with torch.no_grad():
    stego_img = netG(z).detach().cpu()
utils.save_image(stego_img, os.path.join(save_dir, "new_stego_image.png"), normalize=True)

print(f"Image generated and saved to {os.path.join(save_dir, 'new_stego_image.png')}")
