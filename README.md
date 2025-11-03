# AI-Based Watermelon Ripeness Detection in Smart Farming

This project implements various AI techniques for watermelon ripeness detection and analysis in smart farming applications.

## Project Components

### 1. Object Detection (YOLO)
- Custom trained YOLOv8 model for watermelon detection
- Dataset with labeled watermelon images
- Training and prediction scripts

### 2. Classification
- Watermelon ripeness classification model
- Multiple trained models (`.pth` files)
- Jupyter notebooks for training and testing

### 3. GAN (Generative Adversarial Network)
- Generator and Discriminator models
- Watermelon image generation capabilities
- Training notebook included

### 4. Segmentation
- Watermelon segmentation model
- CSV files for data organization
- Training and testing notebooks

### 5. Image Processing
- Image flipping utility
- Data preprocessing tools

### 6. Stable Diffusion
- Fine-tuned Stable Diffusion model
- Training notebook for custom watermelon generation

## Environment Setup
- Multiple Python environments are included for different components
- Requirements can be found in respective environment folders

## Dataset
- Organized dataset structure with images and labels
- Multiple data splits (train/val) for different models
- Annotated data for object detection and segmentation

## Models
Pre-trained models included:
- YOLOv8s for object detection
- Classification models
- Segmentation models
- GAN models
- Fine-tuned Stable Diffusion model

## Usage
Please refer to individual component directories for specific usage instructions and requirements.

## Project Structure
```
├── GAN/                      # Generative Adversarial Network
├── Object_Detection/         # YOLO-based detection
├── project_classification/   # Ripeness classification
├── project_flipping/        # Image processing
├── segmentation/            # Semantic segmentation
└── StableDiffusionModel/    # Stable Diffusion
```