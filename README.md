# Image Upscaler API

GPU-accelerated image upscaling via a FastAPI REST service, using TensorRT-optimized ESRGAN models. Supports 1×, 2×, and 4× upscaling with automatic model scale detection.

> Inspired by [ComfyUI-Upscaler-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt)

## Quick Start (Docker)

```bash
docker compose up --build
```

On first launch the entrypoint downloads ONNX models from HuggingFace, builds TensorRT engines, then starts the API. Subsequent starts skip the build.

The API is at `http://localhost:8123` — Swagger UI at `/docs`.

## Endpoints

### `POST /upscale`

Upload an image for upscaling.

| Form field   | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `image`     | file   | —       | Input image (256–1280 px) |
| `resize_to` | string | `none`  | Target resolution: `none` (model-native), `HD`, `FHD`, `2k`, `4k`, or a multiplier like `2x`, `3.5x` |
| `model_name`| string | default  | Stem of a `.trt` file in the models directory |

Returns `image/png`.

The model's native scale factor is parsed from its filename (e.g. `4x-UltraSharp` → 4×, `2x-ESRGAN` → 2×).

### `GET /models`

List available compiled models.

### `GET /`

Redirects to `/docs`.

## Configuration (`.env`)

| Variable              | Default | Description           |
|-----------------------|---------|-----------------------|
| `API_HOST`            | `0.0.0.0` | Bind address       |
| `API_PORT`            | `8123`    | Bind port          |
| `WORKER_NUM`          | `2`       | Uvicorn workers    |
| `MODEL_PATH`          | `./models`| TRT engines directory |
| `HF_TOKEN`            | —         | HuggingFace token (unauthenticated works but may be rate-limited) |


## Models

14 ESRGAN variants auto-downloaded from [yuvraj108c/ComfyUI-Upscaler-Onnx](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx):

**4×**
- `4x_UniversalUpscalerV2-Neutral_115000_swaG` (default)
- `4x-UltraSharp`, `4x-UltraSharpV2`, `4x-UltraSharpV2_Lite`
- `RealESRGAN_x4`
- `4x-AnimeSharp`, `4x-ClearRealityV1`
- `4x_foolhardy_Remacri`, `4x_RealisticRescaler_100000_G`
- `4x_NMKD-Siax_200k`, `4x-WTP-UDS-Esrgan`, `4xNomos2_otf_esrgan`

**2×**
- `2x-ESRGAN`

**1×**
- `1x-ITF-SkinDiffDetail-Lite-v1`

## Requirements

- NVIDIA GPU with CUDA 12.1+
- Docker + nvidia-container-toolkit (for Docker deployment)
- Python 3.10+, PyTorch 2.3, TensorRT (for manual deployment)
