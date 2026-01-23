# SatGen-CV

**Semantic Image Synthesis for Satellite Imagery from Road and Building Segmentation Masks**

A Pix2Pix-based system for generating realistic satellite imagery from semantic segmentation masks containing roads and buildings. This project addresses the challenge of generating synthetic large-scale aerial imagery.

## Web Interface

Want to try SatGen-CV without setting up the full environment? Check out our interactive web interface: **[satgen-web](https://github.com/Guereak/roadgen-web)**

## Features

- **Pix2Pix Pipeline**: Complete image-to-image translation pipeline specialized for satellite image synthesis from two-channel segmentation masks (roads and buildings)
- **Seamless Generation**: Cosine-fade blending method that produces arbitrarily large images without visible patch boundaries
- **SAM Integration**: Automated building/road mask generation using Segment Anything Model (SAM) for working with partially labeled datasets
- **Multi-Layer Perceptual Loss**: VGG-19 based perceptual loss for improved texture quality


## Installation / Usage

Requires Python 3.8+. Was tested on Python 3.12

```bash
git clone https://github.com/Guereak/SatGen-CV.git
cd SatGen-CV
pip install -r requirements.txt
```

### Pretrained Model

Download the pretrained model from Hugging Face before running inference:

```bash
huggingface-cli download Guereak/SatGen-CV --local-dir ./checkpoints
```

Or manually download from: https://huggingface.co/Guereak/SatGen-CV

### Preprocessed Dataset

You can download the preprocessed dataset (with SAM-generated masks) from Google Cloud Storage:

```bash
gsutil -m cp -r gs://roadgen-cv/data ./
```

This includes train/test/validation splits with 256x256 patches ready for training.


If you have downloaded the datasets yourself (sources in the references section), you must preprocess them using:

```bash
./scripts/preprocess.sh
```

And generate the Masks from Sam3 using SAM3:

```bash
./scripts/sam3_inference.sh
```



### Training

If you have your dataset ready and in the right format, you can train the model with

```bash
./scripts/train.sh
```



#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 1-8 |
| Epochs | 10 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Learning rate | 0.0002-0.0003 |
| LR scheduler | CosineAnnealing (η_min=1e-6) |
| λ_GAN | 1.0 |
| λ_L1 | 10 |
| λ_perc | 50 |
| Patch size | 256x256 |

### Inference via API

If you use this project with roadgen-web (or need an API), you can use our very basic FastAPI implementation:

```bash
python api.py
```



## Architecture

### Generator
- U-Net architecture with skip connections
- 8 downsampling/upsampling blocks
- Instance Normalization + LeakyReLU (encoder) / ReLU (decoder)
- ~54.4M parameters
- Input: 256x256x2 (road + building masks)
- Output: 256x256x3 (RGB satellite image)

### Discriminator
- PatchGAN architecture
- Classifies overlapping image patches as real/fake
- Input: concatenated segmentation mask and image (HxWx5)

### Loss Functions
```
L_G = λ_GAN * L_GAN + λ_L1 * L_L1 + λ_perc * L_perc
```
- **Adversarial Loss**: Binary cross-entropy with label smoothing
- **L1 Reconstruction Loss**: Pixel-wise consistency
- **Perceptual Loss**: VGG-19 features

## Datasets Used

### Massachusetts Roads Dataset
- 1,171 aerial images at 1500x1500 pixels
- Binary road segmentation masks from OpenStreetMap

### Inria Aerial Image Labeling Dataset
- High-resolution imagery from Austin, Chicago, Kitsap County, Western Tyrol, and Vienna
- 5000x5000 pixels (rescaled to 1500x1500)
- Binary building footprint annotations

Both datasets are augmented with SAM-generated masks to create complementary two-channel inputs.


### Seamless Generation

The seamless generation pipeline uses:
- Patch size: 256 pixels
- Overlap: 128 pixels (50%)
- Cosine-fade blending for smooth transitions
- Noise injection for varied background textures on empty patches
- Inference-time dropout for output diversity

## References

- [Pix2Pix](https://arxiv.org/abs/1611.07004) - Image-to-Image Translation with Conditional Adversarial Networks
- [SAM3](https://github.com/facebookresearch/sam3) - Segment Anything Model v3
- [Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)
- [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/)
