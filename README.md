# Image Upscaler API

GPU-accelerated 4× image upscaling via a FastAPI REST service, using TensorRT-optimized ESRGAN models.

> Inspired by [ComfyUI-Upscaler-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt)

## Quick Start (Docker)

```bash
docker compose up --build
```

On first launch it downloads ONNX models from HuggingFace, builds TensorRT engines, then starts the API. Subsequent starts skip the build.

The API is at `http://localhost:8123` — Swagger UI at `/docs`.

## Endpoints

### `POST /upscale`

Upload an image for 4× upscaling.

| Form field   | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `image`     | file   | —       | Input image (256–1280 px) |
| `resize_to` | string | `none`  | Target resolution: `none` (4×), `HD`, `FHD`, `2k`, `4k`, or a multiplier like `2x`, `3.5x` |
| `model_name`| string | default  | Stem of a `.trt` file in the models directory |

Returns `image/png`.

### `GET /models`

List available compiled models.

### `GET /`

Redirects to `/docs`.

## Manual Setup

```bash
python -m venv venv-upscaler
source venv-upscaler/bin/activate
pip install -r requirements.txt
```

### Build engines

Place ONNX models in `models/onnx/` (or let the script download them), then:

```bash
python scripts/build_engine.py
```

### Run

```bash
python source/uvicorn.main.py --log-level info
```

## Configuration (`.env`)

| Variable              | Default | Description           |
|-----------------------|---------|-----------------------|
| `API_HOST`            | `0.0.0.0` | Bind address       |
| `API_PORT`            | `8123`    | Bind port          |
| `WORKER_NUM`          | `2`       | Uvicorn workers    |
| `MODEL_PATH`          | `./models`| TRT engines directory |

## Models

12 pre-compilable 4× ESRGAN variants from [yuvraj108c/ComfyUI-Upscaler-Onnx](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx):

- `4x_UniversalUpscalerV2-Neutral_115000_swaG` (default)
- `4x-UltraSharp`, `4x-UltraSharpV2`, `4x-UltraSharpV2_Lite`
- `RealESRGAN_x4`
- `4x-AnimeSharp`
- `4x-ClearRealityV1`
- `4x_foolhardy_Remacri`
- `4x_RealisticRescaler_100000_G`
- `4x_NMKD-Siax_200k`
- `4x-WTP-UDS-Esrgan`
- `4xNomos2_otf_esrgan`

## Requirements

- NVIDIA GPU with CUDA 12
- Docker + nvidia-container-toolkit (for Docker deployment)
- Python 3.11, PyTorch 2.3, TensorRT (for manual deployment)
