#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import shlex
import subprocess
import tarfile

if os.getenv('SYSTEM') == 'spaces':
    subprocess.run(
        shlex.split(
            'pip install git+https://github.com/facebookresearch/detectron2@v0.6'
        ))
    subprocess.run(
        shlex.split(
            'pip install git+https://github.com/aim-uofa/AdelaiDet@7bf9d87'))

import gradio as gr
import huggingface_hub
import numpy as np
import torch
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

DESCRIPTION = '# [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter)'

MODEL_REPO = 'public-data/Yet-Another-Anime-Segmenter'


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset')
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(device: torch.device) -> DefaultPredictor:
    config_path = huggingface_hub.hf_hub_download(MODEL_REPO, 'SOLOv2.yaml')
    model_path = huggingface_hub.hf_hub_download(MODEL_REPO, 'SOLOv2.pth')
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device.type
    cfg.freeze()
    return DefaultPredictor(cfg)


def predict(image_path: str, class_score_threshold: float,
            mask_score_threshold: float,
            model: DefaultPredictor) -> tuple[np.ndarray, np.ndarray]:
    model.score_threshold = class_score_threshold
    model.mask_threshold = mask_score_threshold
    image = read_image(image_path, format='BGR')
    preds = model(image)
    instances = preds['instances'].to('cpu')

    visualizer = Visualizer(image[:, :, ::-1])
    vis = visualizer.draw_instance_predictions(predictions=instances)
    vis = vis.get_image()

    masked = image.copy()[:, :, ::-1]
    mask = instances.pred_masks.cpu().numpy().astype(int).max(axis=0)
    masked[mask == 0] = 255

    return vis, masked


image_paths = load_sample_image_paths()
examples = [[path.as_posix(), 0.1, 0.5] for path in image_paths]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)

fn = functools.partial(predict, model=model)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input', type='filepath')
            class_score_threshold = gr.Slider(label='Score Threshold',
                                              minimum=0,
                                              maximum=1,
                                              step=0.05,
                                              value=0.1)
            mask_score_threshold = gr.Slider(label='Mask Score Threshold',
                                             minimum=0,
                                             maximum=1,
                                             step=0.05,
                                             value=0.5)
            run_button = gr.Button('Run')
        with gr.Column():
            result_instances = gr.Image(label='Instances')
            result_masked = gr.Image(label='Masked')

    inputs = [image, class_score_threshold, mask_score_threshold]
    outputs = [result_instances, result_masked]
    gr.Examples(examples=examples,
                inputs=inputs,
                outputs=outputs,
                fn=fn,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')
    run_button.click(fn=fn, inputs=inputs, outputs=outputs, api_name='predict')
demo.queue(max_size=15).launch()
