# Deep Convolutional GAN-Based Image Steganography with Decoder

This project implements a **deep learning-based image steganography system** using a **DCGAN-style architecture** to encode and decode hidden messages within generated face images from the CelebA dataset.

Originally, the goal was to find a way to implement a steganographic system that is photo-resilient, however, through analysis of different techniques I have arrived at a DCGAN-style architecture that lays the grounds for eventually making a system that is photo-resilient.

---

## Project Overview

I trained a **Generator (G)** to convert a latent vector `z` into realistic face images, while simultaneously training a **Decoder (Dᵤ)** to reverse-engineer that image and reconstruct `z`. By embedding a binary-encoded message directly into the latent space, the system generates an image that *is the message*. No additional payload is hidden—**the entire image represents the message**.

---

## Directory Structure

```
project/
│
├── dcgan_based_trainer_v2_4.py               # Main training script
├── decode_image_v1_6.py                # Decoding-only script (no retraining needed)
├── output/
│   ├── netG.pth             # Trained generator
│   ├── netDec.pth           # Trained decoder
│   ├── stego_result.png     # Image generated from a message
│   └── epoch_XXX.png        # Sample outputs per epoch
├── img_align_celeba/        # CelebA dataset (downloaded separately)
```

---

## How It Works

1. **Message Encoding:**
   - A text message (e.g., `"hello world"`) is converted into a binary string.
   - Each bit is mapped to a float: `1 → +1.0`, `0 → -1.0`, forming the latent vector `z`.

2. **Image Generation:**
   - The generator `G(z)` maps `z` into a 64x64 RGB image resembling a face.
   - This image is saved as `stego_result.png`.

3. **Message Decoding:**
   - The decoder `Dᵤ(img)` extracts a predicted `ẑ` from the image.
   - Bit-by-bit comparison reconstructs the original message.

4. **Bit Accuracy:**
   - The system reports the % of bits correctly recovered from the image.
   - This gives insight into the fidelity and robustness of the encoding.

---

## Models

- **Generator (G):** Deep Convolutional Generator (DCGAN) that upsamples from latent space.
- **Discriminator (optional for training):** Standard DCGAN discriminator.
- **Decoder (Dᵤ):** CNN that recovers the original latent vector from the generated image.

---

## Testing the Decoder Only

To run the decoder without retraining:

```bash
python decode.py
```

Make sure the following files exist:
- `output/netG.pth`
- `output/netDec.pth`
- `output/stego_result.png`

---

## Example Output

```
Original message:  hello world
Decoded message:  hello world
Bit accuracy:     98.75%
```

## My tests

Here is the "natural" image i generated with the string "Hello World"

![stego_result](https://github.com/user-attachments/assets/dbfa148c-def4-4539-a9e7-de7b7a1221db)

After running the decoder we are successfully able to get the message.
![image](https://github.com/user-attachments/assets/30193914-4b56-4e7e-8af4-143161646004)


---

## 📝 Notes

- I used **MSE Loss** for reconstruction and **BCELoss** for adversarial training.
- `lambda_recon` balances realism vs latent recovery. Higher values prioritize message integrity.

---


