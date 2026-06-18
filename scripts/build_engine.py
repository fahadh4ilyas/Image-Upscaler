import os, sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, 'source'))

from api.tools import Engine

HF_REPO = "yuvraj108c/ComfyUI-Upscaler-Onnx"


def download_onnx(onnx_dir):
    os.makedirs(onnx_dir, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns="*.onnx",
        local_dir=onnx_dir,
        local_dir_use_symlinks=False,
    )


onnx_dir = os.path.join(ROOT_DIR, 'models', 'onnx')
trt_dir = os.path.join(ROOT_DIR, 'models')

existing = list(Path(onnx_dir).glob("*.onnx"))
if not existing:
    print("No ONNX models found, downloading from HuggingFace …")
    download_onnx(onnx_dir)

model_list = [s.as_posix() for s in Path(onnx_dir).glob("*.onnx")]

print(f"\nBuilding {len(model_list)} TensorRT engine(s) …")

for i, m in enumerate(model_list, 1):
    filename = os.path.basename(m).rsplit(".", 1)[0]
    print(f"  [{i}/{len(model_list)}] {filename} …", flush=True)
    engine = Engine(os.path.join(trt_dir, filename + '.trt'))
    ret = engine.build(onnx_path=m, fp16=True)
    if ret != 0:
        print(f"    FAILED (exit code {ret})")
    else:
        print(f"    done.")

print("Build complete.")