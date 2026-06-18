import os
import traceback
import timeit
import torch
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse, Response, RedirectResponse

from contextlib import asynccontextmanager

from .config import config, LOGGER_ACCESS, LOGGER
from .tools import upscale_image, Engine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- model placeholders (load your model in startup) ---
engine: Optional[Engine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    engine = Engine(os.path.join(REPO_ROOT, config.model_path, config.default_model + '.trt'))
    engine.load()
    LOGGER.info("Startup: model loaded — %s", config.default_model)
    yield
    del engine
    LOGGER.info("Shutdown: clean up resources if needed")

app = FastAPI(title='Image Upscaler API',
    description='API for upscaling image using AI models.',
    version='1.0.0',
    lifespan=lifespan)


@app.exception_handler(Exception)
async def value_error_handler(request: Request, exc: Exception):
    return ORJSONResponse({
        'error': str(exc),
        'traceback': "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        'status_code': 500
    }, status_code=500)


@app.middleware("http")
async def logging_request(request: Request, call_next):

    client_data = ''
    if request.client:
        client_data = f'{request.client.host}:{request.client.port}'
    LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" START')
    params = str(request.query_params)
    body = await request.body()
    if params:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" PARAMS: {params}')
    if body:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" BODY: {body[:256]}')

    start = timeit.default_timer()
    request.state.is_disconnected = request.is_disconnected
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = f'{timeit.default_timer() - start:.6f}'

    return response


@app.post("/upscale")
def upscale_image_endpoint(
    image: UploadFile = File(...),
    resize_to: str = Form('none'),
    model_name: str = Form(config.default_model)
):
    global engine
    
    model_path = os.path.join(REPO_ROOT, config.model_path, model_name + '.trt')
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=400, detail=f'Model "{model_name}" not found')
    if engine.engine_path != model_path:
        engine.reset(model_path)

    image_result = upscale_image(engine, image, resize_to)
    torch.cuda.empty_cache()  # clear cache after inference

    return Response(content=image_result, media_type='image/png')

@app.get('/models')
def list_models():
    model_files = [f.stem for f in Path(os.path.join(REPO_ROOT, config.model_path)).glob('*.trt')]
    return {'models': model_files}

@app.get('/', include_in_schema=False)
async def redirect():

    return RedirectResponse(app.root_path+'/docs')