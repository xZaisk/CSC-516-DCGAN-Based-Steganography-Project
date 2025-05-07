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

## Step by Step

1. Download the CelebA Dataset (aligned)
 - https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 - Extract and place folder in project dir

2. Place All Files in the Same Directory
 - Make sure all py files are in the same dir
 - `dcgan_based_trainer_v2_4.py`
 - `image_generator.py`
 - `decode_image_v1_6.py`
 - `img_align_celeba` (contains all the images)

3. Train the Model
 - `python dcgan_based_trainer_v2_4.py`
 - This trains the generator, decoder, and discriminator using the CelebA dataset.
 - After training, it saves `output/netG.pth`, `output/netDec.pth`, and up to 30 epochs`output/epoch_30.png` (you can change via `num_epochs = 30`
 - It does a test that generates an image based on the string 'hello world' and attempts to decode it

4. Generate a New Image with a Message
 - You can make your own image based on your own message with `image_generator.py`
 - Enter a text message (max ~12–13 characters).

5. Decode a Message from an Image
 - `python decode_image_v1_6.py`
 -  enter the name of an image file
 -  The decoder will extract and print the message


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

## Message Encoding & Decoding Tests

### **Test 1 – Message: `"Hello World"`**
- **Purpose:** Evaluate fidelity of image-based message recovery during initial training.
- **Input Message:**  
  `Hello World`

- **Generated Image:**  
  ![stego_result](https://github.com/user-attachments/assets/dbfa148c-def4-4539-a9e7-de7b7a1221db)

- **Decoded Output:**  
  ![image](https://github.com/user-attachments/assets/30193914-4b56-4e7e-8af4-143161646004)

> The model successfully reconstructed the message with high bit accuracy.

---

### **Test 2 – Message: `"Andrew says hello"`**
- **Purpose:** Test message length near or above capacity limits.
- **Input Message:**  
  `Andrew says hello`

- **Generated Image:**  
  ![image](https://github.com/user-attachments/assets/f073deb1-d356-4c90-a5b7-7c155b1a33a8)

- **Decoded Output:**  
  ![image](https://github.com/user-attachments/assets/fcc83d22-48cb-4ee6-8d9c-e5cb7c86a3bb)

>  The decoder recovered a mostly correct message. Slight truncation or distortion suggests room for improvement in message capacity and decoding accuracy.


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
