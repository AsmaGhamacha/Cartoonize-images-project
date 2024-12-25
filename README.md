# Cartoonize Images with Pix2Pix GAN

## Overview
This project uses a Pix2Pix Generative Adversarial Network (GAN) to transform real-world images into cartoonized versions. The steps include dataset preparation, model training, validation, and evaluation to achieve this image-to-image transformation.

## Dataset
- **Dataset Used**: [Unsplash Lite 5K Dataset - Color Subset](https://www.kaggle.com/datasets/matthewjansen/unsplash-lite-5k-colorization)
- **Training Dataset**: 3000 images
- **Testing Dataset**: 500 images (validation set)

## Methodology
### **1. Dataset Preparation**
- **Data Split**: The dataset was split into training and testing sets.
- **Cartoonization Techniques**:
  - Applied methods based on:
    - Non-Photorealistic Rendering (source: LearnOpenCV).
    - OpenCV cartoonization pipeline (source: [GitHub project](https://github.com/Shaashwat05/Cartoonify_reality)).
    - Image transformations and filtering techniques (source: IoT4Beginners blog).

### **2. Data Preparation**
- Paired the real-world images and their cartoonized counterparts using `prepare_data.py`.
- Output: A dataset ready for Pix2Pix training with paired images (real and cartoonized).

### **3. Model Training**
- **Model**: Pix2Pix GAN, implemented in `pix2pix_model.py`.
- **Training Parameters**:
  - **Epochs**: 50 (saved checkpoints every 5 epochs).
  - **Batch Size**: 16.
  - **Loss Function**: Combined L1 loss (for reconstruction) and adversarial loss.
  - **Optimizer**: Adam with a learning rate of 0.0002.
- **Total Training Time**: 4 hours, 7 minutes, and 54 seconds.

### **4. Validation**
- Validated the model on 500 unseen test images.
- Generated output images showing the input, output (generated), and target images. 
- Example outputs:

| **Input Image**            | **Output Image**            | **Target Image**            |
|-----------------------------|-----------------------------|-----------------------------|
| ![Input](![input_303](https://github.com/user-attachments/assets/8507c3ef-ca5c-4ba7-a670-3cd6000b216c)
) | ![Output](![output_303](https://github.com/user-attachments/assets/c27bb4e7-bfce-46d5-80f3-7c330007ff95)
) | ![Target](![target_303](https://github.com/user-attachments/assets/e981698f-5091-451c-985e-f1227452cd33)
) |
| ![Input](![input_283](https://github.com/user-attachments/assets/c0812b5f-1d49-4a73-a1bb-d485feacf9be)
) | ![Output](![output_283](https://github.com/user-attachments/assets/cbd5012c-0bb3-4039-932f-0b26fb0181e3)
) | ![Target](![target_283](https://github.com/user-attachments/assets/1f0d15dc-27f3-466f-91dc-079bdee31d4a)
) |
| ![Input](![input_492](https://github.com/user-attachments/assets/762e45b3-c213-4cf7-b2d2-cdf1b0e61c6d)
) | ![Output](![output_492](https://github.com/user-attachments/assets/3f6aa2f0-8a54-4935-b75e-d053d0fa87ef)
) | ![Target](![target_492](https://github.com/user-attachments/assets/8b04ef0b-64ee-4ea7-9ab6-ef8f2b8a3d7f)
) |

### **5. Evaluation**
- **Performance Metrics**:
  - Average Structural Similarity Index (SSIM): **0.7986** (calculated for all test pairs).
- **Plots**:
  - Loss vs. Epochs:
    ![Loss vs Epochs](![target_492](https://github.com/user-attachments/assets/ce404097-07eb-4f55-88e3-f5d288f6c9b0)
)
  - Training Time vs. Epochs:
    ![Training Time vs Epochs](![target_492](https://github.com/user-attachments/assets/20bb1708-21d9-4b43-8b0a-3c80ef22ee1e)
)