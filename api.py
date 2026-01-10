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
        noise_kwargs={"noise_scale": 10.0},
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

    # TODO: Set your checkpoint path here or via environment variable
    checkpoint_path = "checkpoints/best_model.pth"

    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Model will be loaded on first request if checkpoint path is provided")
        return

    print(f"Loading model from {checkpoint_path}...")
    generator = load_model(checkpoint_path, device)
    print("Model loaded successfully!")


def preprocess_image(
        image_bytes: bytes,
        filename: str,
        c1_pixel_value: Tuple[int, int, int] = (255, 0, 0),
        c2_pixel_value: Tuple[int, int, int] = (0, 255, 0),
        save_debug: bool = True
    ) -> np.ndarray:
    """ Convert uploaded image to the expected format (H, W, 2) for the model. """
    # Read image
    if filename.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(io.BytesIO(image_bytes))
    else:
        image = np.array(Image.open(io.BytesIO(image_bytes)))


   # Handle both RGB (3 channels) and RGBA (4 channels)
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got shape: {image.shape}")

    if image.shape[2] == 4:
        image = image[:, :, :3]

    mask = np.zeros((image.shape[0], image.shape[1], 2))

    mask[:, :, 0] = (image == c1_pixel_value).all(axis=2) * 255
    mask[:, :, 1] = (image == c2_pixel_value).all(axis=2) * 255

    # Save mask for debugging
    if save_debug:
        debug_dir = Path("debug_masks")
        debug_dir.mkdir(exist_ok=True)

        # Save as TIFF (2-channel)
        tifffile.imwrite(debug_dir / "mask_2channel.tif", mask.astype(np.uint8))

        # Save each channel as separate PNG for easy viewing
        Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).save(debug_dir / "channel_0_buildings.png")
        Image.fromarray((mask[:, :, 1] * 255).astype(np.uint8)).save(debug_dir / "channel_1_roads.png")

        # Save a combined RGB visualization (Red=buildings, Green=roads)
        combined = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        combined[:, :, 0] = mask[:, :, 0] * 255  # Red channel = buildings
        combined[:, :, 1] = mask[:, :, 1] * 255  # Green channel = roads
        Image.fromarray(combined).save(debug_dir / "combined_visualization.png")

        print(f"Debug masks saved to {debug_dir}")

    return mask


@app.post("/generate",
          summary="Generate satellite image",
          description="Upload a segmentation mask and get a generated satellite image")
async def generate_image(
    file: UploadFile = File(..., description="Segmentation mask (TIFF, PNG, or JPG)"),
    # patch_size: int = Query(256, description="Size of patches for processing"),
    # overlap: int = Query(64, description="Overlap between patches"),
    # blend_mode: str = Query("cosine", description="Blending mode: linear or cosine"),
    # noise_strategy: str = Query("gaussian", description="Noise strategy: gaussian, sparse, uniform, per_channel, none"),
    # noise_scale: float = Query(10.0, description="Scale/strength of noise for empty patches"),
    # empty_threshold: float = Query(0.01, description="Threshold for considering patch empty (0.0-1.0)"),
    # enable_dropout: bool = Query(True, description="Keep dropout enabled during inference (recommended for Pix2Pix)"),
    # checkpoint: Optional[str] = Query(None, description="Path to model checkpoint (optional)")
):
    """
    Generate a seamless satellite image from a segmentation mask.

    **Input format:**
    - TIFF with 2 channels: buildings (channel 0) and roads (channel 1)
    - RGB image: R=buildings, G=roads
    - Grayscale: assumes roads only

    **Returns:** PNG image
    """
    global generator

    # Load model if not already loaded or if different checkpoint specified
    if generator is None or checkpoint is not None:
        if checkpoint is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please provide a checkpoint path."
            )

        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"

        try:
            generator = load_model(checkpoint, device)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )

    # Read and preprocess the uploaded image
    try:
        image_bytes = await file.read()
        mask = preprocess_image(image_bytes, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process input image: {str(e)}"
        )

    # Update generator parameters if different from defaults
    needs_update = (
        patch_size != generator.patch_size or
        overlap != generator.overlap or
        blend_mode != generator.blend_mode or
        empty_threshold != generator.empty_threshold or
        enable_dropout != generator.enable_dropout
    )

    if needs_update:
        generator.patch_size = patch_size
        generator.overlap = overlap
        generator.blend_mode = blend_mode
        generator.empty_threshold = empty_threshold
        generator.enable_dropout = enable_dropout
        generator.weights = create_blend_weights(patch_size, overlap, blend_mode)

        # Update noise strategy
        from src.inference.noise_strategies import create_noise_strategy
        noise_kwargs = {}
        if noise_strategy == "gaussian":
            noise_kwargs = {"noise_scale": noise_scale}
        elif noise_strategy == "sparse":
            noise_kwargs = {"density": noise_scale / 1000.0}
        elif noise_strategy == "uniform":
            noise_kwargs = {"noise_range": noise_scale}
        elif noise_strategy == "per_channel":
            noise_kwargs = {"noise_scales": [noise_scale, noise_scale]}

        generator.noise_strategy = create_noise_strategy(noise_strategy, **noise_kwargs)

        # Re-configure dropout if needed
        if enable_dropout:
            generator._enable_pix2pix_dropout()
        else:
            generator.model.eval()

    # Generate the image
    try:
        result = generator.generate(mask)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

    # Convert result to PNG and return
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
