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
├── decode_image_v1_6.py                      # Decoding-only script (no retraining needed)
├── image_generator.py                        # generates an image based on your string
├── output/
│   ├── netG.pth             # Trained generator
│   ├── netDec.pth           # Trained decoder
│   ├── stego_result.png     # Image generated from a message
│   └── epoch_XXX.png        # Sample outputs per epoch
├── img_align_celeba/        # CelebA dataset (downloaded separately)
```

---

## How It Works

0. **Datasets**
   - I used the aligned and cropped photos from CelebA Datasets
   - https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. **Message Encoding:**
   - A text message (e.g., `"hello world"`) is converted into a binary string.
   - Each bit is mapped to a float: `1 → +1.0`, `0 → -1.0`, forming the latent vector `z`.

3. **Image Generation:**
   - The generator `G(z)` maps `z` into a 64x64 RGB image resembling a face.
   - This image is saved as `stego_result.png`.

4. **Message Decoding:**
   - The decoder `Dᵤ(img)` extracts a predicted `ẑ` from the image.
   - Bit-by-bit comparison reconstructs the original message.

5. **Bit Accuracy:**
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

Here is the "natural" image i generated with the string "Hello World" when first training the mode

![stego_result](https://github.com/user-attachments/assets/dbfa148c-def4-4539-a9e7-de7b7a1221db)

After running the decoder we are successfully able to get the message.
![image](https://github.com/user-attachments/assets/30193914-4b56-4e7e-8af4-143161646004)


I generated another image with the message "Andrew says hello":
![image](https://github.com/user-attachments/assets/f073deb1-d356-4c90-a5b7-7c155b1a33a8)

Here is the output of the decoder:
![image](https://github.com/user-attachments/assets/fcc83d22-48cb-4ee6-8d9c-e5cb7c86a3bb)

It performs pretty well, although there can be an improvement. 
---

## Notes

- I used **MSE Loss** for reconstruction and **BCELoss** for adversarial training.
- `lambda_recon` balances realism vs latent recovery. Higher values prioritize message integrity.

---

## Discussion
This project demonstrates that it's possible to encode and recover messages directly from synthetic images with impressive accuracy, leveraging DCGAN architecture. Rather than hiding information in existing cover images, the image itself is the message—making it an inherently stealthy approach.

### Limitations:
- Image fragility: Small distortions (e.g., compression, screenshots, or social media filters) may significantly degrade message accuracy.
- Message length: Limited by the size of the latent vector z (currently 100 bits ≈ 12–13 characters).
- Domain-specific generation: Since training uses CelebA, all generated images are human faces—making it less flexible for general-purpose use.

### Possible Improvements:
- **Error correction:** Integrate bit redundancy or ECC (e.g., Hamming codes) to recover messages with higher tolerance.
- Larger or variable-length encoding: Train with dynamic latent vector sizes or multi-image message splitting.
- **Transfer to natural steganography:** Use the decoder on screenshots of generated images to test photo-resilience.
