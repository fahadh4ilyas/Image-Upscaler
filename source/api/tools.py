import torch, io
from torch.cuda import nvtx
from fastapi import UploadFile
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
from logging import error
import copy

G_LOGGER.module_severity = G_LOGGER.ERROR

IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 1280

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

def get_final_resolutions(width, height, resize_to):
    final_width = None
    final_height = None
    aspect_ratio = float(width/height)

    match resize_to:
        case "HD":
            final_width = 1280
            final_height = 720
        case "FHD":
            final_width = 1920
            final_height = 1080
        case "2k":
            final_width = 2560
            final_height = 1440
        case "4k":
            final_width = 3840
            final_height = 2160
        case "none":
            final_width = width*4
            final_height = height*4

        case _:
            resize_factor = float(resize_to.split('x')[0])
            final_width = width*resize_factor
            final_height = height*resize_factor

    if aspect_ratio == 1.0:
        final_width = final_height

    if aspect_ratio < 1.0 and resize_to not in ("none", "1x", "1.5x", "2x", "2.5x", "3x", "3.5x", "4x", "5x", "6x", "7x", "8x", "9x", "10x"):
        temp = final_width
        final_width = final_height
        final_height = temp

    return (int(final_width), int(final_height)) # must be whole numbers

class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path = None):

        del self.context
        del self.buffers
        del self.tensors
        if engine_path:
            del self.engine
            self.engine_path = engine_path
            self.load()

        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(
        self,
        onnx_path: str,
        fp16: bool
    ):

        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX

        input_profile = [
            {"input": [(engine_min_batch,engine_channel,engine_min_h,engine_min_w), (engine_opt_batch,engine_channel,engine_opt_h,engine_opt_w), (engine_max_batch,engine_channel,engine_max_h,engine_max_w)]},
        ]

        p = [Profile() for i in range(len(input_profile))]
        for _p, i_profile in zip(p, input_profile):
            for name, dims in i_profile.items():
                assert len(dims) == 3
                _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        network = network_from_onnx_path(
            onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
        )

        builder = network[0]
        config = builder.create_builder_config()

        config.set_flag(trt.BuilderFlag.FP16) if fp16 else None

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        try:
            engine = engine_from_network(
                network,
                config,
            )
        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1
        try:
            save_engine(engine, path=self.engine_path)
        except Exception as e:
            error(f"Failed to save engine: {e}")
            return 1
        return 0

    def load(self):
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __str__(self):
        out = ""
            
        # When raising errors in the upscaler, this str() called by comfy's execution.py,
        # but the engine won't have the attributes required for stringification
        # If str() also raises an error, comfy gets soft-locked, not running prompts until restarted
        if not hasattr(self.engine, "num_optimization_profiles") or not hasattr(self.engine, "num_bindings"):
            return out
        
        for opt_profile in range(self.engine.num_optimization_profiles):
            for binding_idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(binding_idx)
                shape = self.engine.get_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out

def upscale_image(engine: Engine, image: UploadFile, resize_to: str, format: str = 'PNG') -> bytes:
    convert_to_tensor = transforms.ToTensor()
    convert_to_pil = transforms.ToPILImage()

    image_bytes = image.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    images_bchw = convert_to_tensor(pil_image).unsqueeze(0)

    B, C, H, W = images_bchw.shape

    for dim in (H, W):
        if dim > IMAGE_DIM_MAX or dim < IMAGE_DIM_MIN:
            raise ValueError(f"Input image dimensions fall outside of the supported range: {IMAGE_DIM_MIN} to {IMAGE_DIM_MAX} px!\nImage dimensions: {W}px by {H}px")
    
    final_width, final_height = get_final_resolutions(W, H, resize_to)

    shape_dict = {
        "input": {"shape": (1, 3, H, W)},
        "output": {"shape": (1, 3, H*4, W*4)},
    }

    engine.activate()
    engine.allocate_buffers(shape_dict=shape_dict)

    cudaStream = torch.cuda.current_stream().cuda_stream
    images_list = list(torch.split(images_bchw, split_size_or_sections=1))
    upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device='cuda')
    must_resize = W*4 != final_width or H*4 != final_height

    for i, img in enumerate(images_list):
        result = engine.infer({"input": img}, cudaStream)
        result = result["output"]

        if must_resize:
            result = torch.nn.functional.interpolate(
                result, 
                size=(final_height, final_width),
                mode='bicubic',
                antialias=True
            )
        upscaled_frames[i] = result
    
    engine.reset()

    final_image = convert_to_pil(upscaled_frames[0].detach().cpu())
    image_byte_arr = io.BytesIO()
    final_image.save(image_byte_arr, format=format)
    final_image = image_byte_arr.getvalue()

    return final_image