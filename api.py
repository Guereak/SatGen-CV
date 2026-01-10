import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tifffile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from src.models.generator import Generator
from src.inference.blending import SeamlessGenerator, create_blend_weights

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Road Generation API",
    description="Generate seamless satellite images from segmentation masks using Pix2Pix",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded at startup)
generator: Optional[SeamlessGenerator] = None


def load_model(checkpoint_path: str, device: str) -> SeamlessGenerator:
    """Load the model and create a SeamlessGenerator instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    in_channels = config.get('input_channels', 2)
    out_channels = config.get('output_channels', 3)
    features = config.get('generator_features', 64)

    # Initialize model
    model = Generator(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['generator_state_dict'])

    # NOTE: Do NOT call model.eval() here for Pix2Pix!
    # The SeamlessGenerator will properly configure dropout and BatchNorm modes

    # Create seamless generator with randomness for empty patches
    seamless_gen = SeamlessGenerator(
        model=model,
        patch_size=256,
        overlap=64,
        blend_mode="cosine",
        device=device,
        noise_strategy="gaussian",  # Add randomness to empty patches
        noise_kwargs={"noise_scale": 10.0, "empty_only": False},
        empty_threshold=0.01,
        enable_dropout=True  # Keep dropout enabled for Pix2Pix
    )

    return seamless_gen


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global generator

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/best_model.pth"

    print(f"Loading model from {checkpoint_path}...")
    generator = load_model(checkpoint_path, device)
    print("Model loaded successfully!")


def preprocess_image(
        image_bytes: bytes,
        filename: str,
        c1_pixel_value: Tuple[int, int, int] = (255, 0, 0),
        c2_pixel_value: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
    """ Convert uploaded image to the expected format (H, W, 2) for the model. """
    # Read image
    if filename.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(io.BytesIO(image_bytes))
    else:
        image = np.array(Image.open(io.BytesIO(image_bytes)))

    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got shape: {image.shape}")

   # Handle both RGB (3 channels) and RGBA (4 channels)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    mask = np.zeros((image.shape[0], image.shape[1], 2))

    mask[:, :, 0] = (image == c1_pixel_value).all(axis=2) * 255
    mask[:, :, 1] = (image == c2_pixel_value).all(axis=2) * 255

    return mask


@app.post("/generate",
          summary="Generate satellite image",
          description="Upload a segmentation mask and get a generated satellite image")
async def generate_image(
    file: UploadFile = File(...),
):
    global generator

    try:
        image_bytes = await file.read()
        mask = preprocess_image(image_bytes, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process input image: {str(e)}"
        )

    try:
        result = generator.generate(mask)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

    output_image = Image.fromarray(result)
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=generated.png"}
    )


@app.get("/health")
async def health_check():
    """Check if the API is running and if the model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": generator is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
