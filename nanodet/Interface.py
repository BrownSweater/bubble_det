#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/26 15:55
# @Author : jj.wang

import cv2
import os
import time
import json
import torch
import argparse
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model config file path')
    parser.add_argument('--model', help='model file path')
    parser.add_argument('--device', type=str, default='cpu', help='gpu or cpu')
    args = parser.parse_args()
    return args


def load_model_weight(model, checkpoint):
    state_dict = checkpoint['state_dict']
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


class Predictor(object):
    def __init__(self, cfg, model_path, score_thresh, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        self.score_thresh = score_thresh
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.cache = {'img': None, 'result': None}

    def __call__(self, img):
        img_info = {}
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
            self._update_cache(meta['raw_img'], results)
            res = self._struct_results(results)
        return res

    def visualize(self, score_thres=None):
        score_thres = score_thres if score_thres is not None else self.score_thresh
        time1 = time.time()
        self.model.head.show_result(self.cache['img'], self.cache['results'], self.cfg.class_names,
                                    score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time() - time1))

    def _update_cache(self, img, results):
        self.cache['img'] = img
        self.cache['results'] = results

    def _struct_results(self, results):
        result = []
        for label in results:
            for index, bbox in enumerate(results[label]):
                score = bbox[-1]
                if score > self.score_thresh:
                    result.append(bbox + [label])
        return result


image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def init_args(config_path=None):
    if config_path is None:
        config_path = os.environ.get('DET_CONFIG_PATH')
    with open(config_path, 'r') as f:
        args = (json.load(f))
    return args


if __name__ != '__main__':
    import platform

    PATH = os.path.basename(os.path.abspath(__file__))
    if platform.system() == 'Darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ['DET_CONFIG_PATH'] = '/Users/wangjunjie/project/nanodet/config.json'
    args = init_args()
    load_config(cfg, args['config'])
    device = 'cuda:0' if args['device'] == 'gpu' else args['device']
    predictor = Predictor(cfg, args['model'], device=device, score_thresh=args['score_thresh'])

if __name__ == '__main__':
    import platform

    if platform.system() == 'Darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ['DET_CONFIG_PATH'] = '/Users/wangjunjie/project/nanodet/config.json'
    args = init_args()
    load_config(cfg, args['config'])
    device = 'cuda:0' if args['device'] == 'gpu' else args['device']
    predictor = Predictor(cfg, args['model'], device=device, score_thresh=args['score_thresh'])
    # x0, y0, x1, y1, score, label_id
    files = get_image_list('VOC2007/JPEGImages')
    for file in files:
        img = cv2.imread(file)
        res = predictor(img)
        predictor.visualize()
        ch = cv2.waitKey(0)
        print(res)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
