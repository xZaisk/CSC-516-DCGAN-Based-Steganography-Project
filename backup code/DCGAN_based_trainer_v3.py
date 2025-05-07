import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from tqdm import tqdm

# ----------------------- CONFIG ----------------------------
data_path = "/img_align_celeba"
save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

image_size = 64
batch_size = 128
nz = 100  # Latent vector size
num_epochs = 30
lr = 0.0002
beta1 = 0.5
lambda_recon = 50.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- CUSTOM DATASET ------------------------
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# ---------------------- TRANSFORM --------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------ MODEL DEFINITIONS ----------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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

# --------------------- MAIN TRAINING -----------------------
if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    dataset = CelebADataset(root_dir=data_path, transform=transform)
    print("Number of images:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netDec = Decoder(nz).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netDec.apply(weights_init)

    criterion = nn.BCELoss()
    recon_loss = nn.MSELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDec = optim.Adam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.
    fake_label = 0.

    print("\nStarting Training...")

    training = False

    if not training:
        for epoch in range(num_epochs):
            for i, data in enumerate(tqdm(dataloader), 0):
                real_images = data.to(device)
                b_size = real_images.size(0)

                # Train Discriminator
                netD.zero_grad()
                label = torch.full((b_size,), real_label, device=device)
                output = netD(real_images).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                optimizerD.step()
                D_G_z1 = output.mean().item()

                # Train Generator + Decoder
                netG.zero_grad()
                netDec.zero_grad()
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                recon = netDec(fake)
                errRecon = recon_loss(recon, noise.view(b_size, -1))
                total_loss = errG + lambda_recon * errRecon
                total_loss.backward()
                optimizerG.step()
                optimizerDec.step()
                D_G_z2 = output.mean().item()

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] [{i}/{len(dataloader)}]  Loss_D: {errD_real + errD_fake:.4f}  Loss_G: {errG:.4f}  Recon: {errRecon:.4f}  D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            utils.save_image(fake, f"{save_dir}/epoch_{epoch+1:03d}.png", normalize=True)

    torch.save(netG.state_dict(), os.path.join(save_dir, "netG.pth"))
    torch.save(netD.state_dict(), os.path.join(save_dir, "netD.pth"))
    torch.save(netDec.state_dict(), os.path.join(save_dir, "netDec.pth"))
    print("\nTraining complete. Models saved.")

    # --------------------- DECODE TEST -----------------------
    def message_to_noise(message, nz=100):
        bits = ''.join(format(ord(c), '08b') for c in message)
        bits = bits[:nz] + '0' * (nz - len(bits))
        z = torch.tensor([1.0 if b == '1' else -1.0 for b in bits], dtype=torch.float32)
        return z.view(1, nz, 1, 1).to(device), bits

    def noise_to_message(decoded_z, threshold=0.0):
        bits = ['1' if val > threshold else '0' for val in decoded_z.squeeze().tolist()]
        chars = [chr(int(''.join(bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
        return ''.join(chars).strip('\x00'), bits

    # Load models for testing
    netG.load_state_dict(torch.load(os.path.join(save_dir, "netG.pth")))
    netDec.load_state_dict(torch.load(os.path.join(save_dir, "netDec.pth")))
    netG.eval()
    netDec.eval()

    test_message = "hello world"
    z_test, original_bits = message_to_noise(test_message, nz)

    with torch.no_grad():
        fake_img = netG(z_test)
        utils.save_image(fake_img, os.path.join(save_dir, "stego_result.png"), normalize=True)
        recovered_z = netDec(fake_img).detach().cpu()

    decoded_msg, decoded_bits = noise_to_message(recovered_z)

    bit_accuracy = sum(o == d for o, d in zip(original_bits, decoded_bits)) / len(original_bits) * 100
    print(f"\nOriginal message:  {test_message}")
    print(f"Decoded message:  {decoded_msg}")
    print(f"Bit accuracy: {bit_accuracy:.2f}%")
