#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import tarfile

try:
    import detectron2
except:
    command = 'pip install git+https://github.com/facebookresearch/detectron2@v0.6'
    subprocess.call(command.split())

try:
    import adet
except:
    command = 'pip install git+https://github.com/aim-uofa/AdelaiDet@7bf9d87'
    subprocess.call(command.split())

import gradio as gr
import huggingface_hub
import numpy as np
import torch
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

ORIGINAL_REPO_URL = 'https://github.com/zymk9/Yet-Another-Anime-Segmenter'
TITLE = 'zymk9/Yet-Another-Anime-Segmenter'
DESCRIPTION = f'A demo for {ORIGINAL_REPO_URL}'
ARTICLE = None

TOKEN = os.environ['TOKEN']
MODEL_REPO = 'hysts/Yet-Another-Anime-Segmenter'
MODEL_FILENAME = 'SOLOv2.pth'
CONFIG_FILENAME = 'SOLOv2.yaml'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--class-score-slider-step', type=float, default=0.05)
    parser.add_argument('--class-score-threshold', type=float, default=0.1)
    parser.add_argument('--mask-score-slider-step', type=float, default=0.05)
    parser.add_argument('--mask-score-threshold', type=float, default=0.5)
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(device: torch.device) -> DefaultPredictor:
    config_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                  CONFIG_FILENAME,
                                                  use_auth_token=TOKEN)
    model_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                 MODEL_FILENAME,
                                                 use_auth_token=TOKEN)
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device.type
    cfg.freeze()
    return DefaultPredictor(cfg)


def predict(image, class_score_threshold: float, mask_score_threshold: float,
            model: DefaultPredictor) -> tuple[np.ndarray, np.ndarray]:
    model.score_threshold = class_score_threshold
    model.mask_threshold = mask_score_threshold
    image = read_image(image.name, format='BGR')
    preds = model(image)
    instances = preds['instances'].to('cpu')

    visualizer = Visualizer(image[:, :, ::-1])
    vis = visualizer.draw_instance_predictions(predictions=instances)
    vis = vis.get_image()

    masked = image.copy()[:, :, ::-1]
    mask = instances.pred_masks.cpu().numpy().astype(int).max(axis=0)
    masked[mask == 0] = 255

    return vis, masked


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    image_paths = load_sample_image_paths()
    examples = [[
        path.as_posix(), args.class_score_threshold, args.mask_score_threshold
    ] for path in image_paths]

    model = load_model(device)

    func = functools.partial(predict, model=model)
    func = functools.update_wrapper(func, predict)

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='file', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=args.class_score_slider_step,
                             default=args.class_score_threshold,
                             label='Class Score Threshold'),
            gr.inputs.Slider(0,
                             1,
                             step=args.mask_score_slider_step,
                             default=args.mask_score_threshold,
                             label='Mask Score Threshold'),
        ],
        [
            gr.outputs.Image(label='Instances'),
            gr.outputs.Image(label='Masked'),
        ],
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
