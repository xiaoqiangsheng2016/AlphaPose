import torch

# import sys
# sys.path.append("./vendor/AlphaPose/")
import cv2
import numpy

from alphapose.models import builder
from trackers.tracker_cfg import cfg as tcfg
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord
from detector.yolo_api import YOLODetector
import detector.yolo_cfg as yolo_cfg
from alphapose.utils.presets import SimpleTransform

from easydict import EasyDict as edict

import glob
import os
import json
import base64
from tqdm import tqdm


class PoseFront(object):
    def __init__(self):
        cfg = update_config("../configs/coco/resnet/kpts_front_256x192_res50_lr1e-3_1x.yaml")
        checkpoint = "../exp/exp_fastpose-kpts_front_256x192_res50_lr1e-3_1x.yaml/model_200.pth"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        self.cfg = cfg

        yolo_cfg.cfg.CONFIG = '../detector/yolo/cfg/yolov3-spp.cfg'
        yolo_cfg.cfg.WEIGHTS = '../detector/yolo/data/yolov3-spp.weights'

        yolo_opt = edict()
        yolo_opt.gpus = [-1]
        yolo_opt.device = "cpu"
        yolo_opt.detbatch = 0
        yolo_opt.posebatch = 0
        yolo_opt.tracking = False
        self.detector = YOLODetector(yolo_cfg.cfg, yolo_opt)

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma,
            train=False, add_dpg=False, gpu_device=self.device)

    def do_detect(self, image_path):
        # img_k = self.detector.image_preprocess(image)
        # self.detector.images_detection(img_k)
        dets_results = self.detector.detect_one_img(image_path)
        if len(dets_results) > 0:
            bbox = dets_results[0]["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        orig_bgr = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        inps, cropped_box = self.transformation.test_transform(orig_img, bbox)

        inps = inps.unsqueeze(0)
        inps = inps.to(self.device)
        func_heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        heatmap = self.pose_model(inps)
        if heatmap.shape[0] > 0:
            heatmap = heatmap[0]
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        preds, maxvals = func_heatmap_to_coord(heatmap, cropped_box, heatmap=hm_size)
        # print(preds)
        # print(maxvals)
        for (x, y) in preds:
            cv2.circle(orig_bgr, center=(int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

        return orig_bgr, preds

class PoseSide(object):
    def __init__(self):
        cfg = update_config("../configs/coco/resnet/kpts_side_256x192_res50_lr1e-3_1x.yaml")
        checkpoint = "../exp/exp_fastpose-kpts_side_256x192_res50_lr1e-3_1x.yaml/model_200.pth"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        self.cfg = cfg

        yolo_cfg.cfg.CONFIG = '../detector/yolo/cfg/yolov3-spp.cfg'
        yolo_cfg.cfg.WEIGHTS = '../detector/yolo/data/yolov3-spp.weights'

        yolo_opt = edict()
        yolo_opt.gpus = [-1]
        yolo_opt.device = "cpu"
        yolo_opt.detbatch = 0
        yolo_opt.posebatch = 0
        yolo_opt.tracking = False
        self.detector = YOLODetector(yolo_cfg.cfg, yolo_opt)

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma,
            train=False, add_dpg=False, gpu_device=self.device)

    def do_detect(self, image_path):
        # img_k = self.detector.image_preprocess(image)
        # self.detector.images_detection(img_k)
        dets_results = self.detector.detect_one_img(image_path)
        if len(dets_results) > 0:
            bbox = dets_results[0]["bbox"]

        # orig_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        orig_bgr = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        inps, cropped_box = self.transformation.test_transform(orig_img, bbox)
        inps = inps.unsqueeze(0)
        inps = inps.to(self.device)
        func_heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        heatmap = self.pose_model(inps)
        if heatmap.shape[0] > 0:
            heatmap = heatmap[0]
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        preds, maxvals = func_heatmap_to_coord(heatmap, cropped_box, heatmap=hm_size)
        # print(preds)
        # print(maxvals)
        for (x, y) in preds:
            cv2.circle(orig_bgr, center=(int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
        # cv2.imwrite("2.jpg", orig_img)
        return orig_bgr, preds

pose_front = PoseFront()
pose_side = PoseSide()

if __name__ == '__main__':
    # image, preds = pose_front.do_detect("./storage/images/1658416958339222600_front.jpg")
    # cv2.imshow("f", image)
    # cv2.waitKey(1000000)

    # image, preds = pose_side.do_detect("./images/1_side.png")
    # cv2.imshow("f",image)
    # cv2.waitKey(1000000)


    data_type = 'side'

    front_labelme_configs = [
        {'name': '颈围', 'index': [0, 1], 'shape_type': 'line'},
        {'name': '前腋宽', 'index': [2, 3], 'shape_type': 'line'},
        {'name': '上胸围', 'index': [4, 5], 'shape_type': 'line'},
        {'name': '下胸围', 'index': [6, 7], 'shape_type': 'line'},
        {'name': '胃围', 'index': [8, 9], 'shape_type': 'line'},
        {'name': '腰围', 'index': [10, 11], 'shape_type': 'line'},
        {'name': '腹围', 'index': [12, 13], 'shape_type': 'line'},
        {'name': '臀围', 'index': [14, 15], 'shape_type': 'line'},
        {'name': '第一大腿围', 'index': [16, 17], 'shape_type': 'line'},
        {'name': '第三大腿围', 'index': [18, 19], 'shape_type': 'line'},
        {'name': '第五大腿围', 'index': [20, 21], 'shape_type': 'line'},
        {'name': '膝上围', 'index': [22, 23], 'shape_type': 'line'},
        {'name': '膝盖围', 'index': [24, 25], 'shape_type': 'line'},
        {'name': '膝下围', 'index': [26, 27], 'shape_type': 'line'},
        {'name': '腿肚围', 'index': [28, 29], 'shape_type': 'line'},
        {'name': '腿肚下围', 'index': [30, 31], 'shape_type': 'line'},
        {'name': '脚腕以上', 'index': [32, 33], 'shape_type': 'line'},
        {'name': '脚腕', 'index': [34, 35], 'shape_type': 'line'},
        {'name': '肩胸长', 'index': [36, 37], 'shape_type': 'line'},
        {'name': '乳长', 'index': [38, 39], 'shape_type': 'line'},
        {'name': '腹长', 'index': [40, 41], 'shape_type': 'line'},
        {'name': '头顶', 'index': [42], 'shape_type': 'point'},
        {'name': 'BP上', 'index': [43], 'shape_type': 'point'},
        {'name': 'BP右', 'index': [44], 'shape_type': 'point'},
        {'name': 'BP左', 'index': [45], 'shape_type': 'point'},
        {'name': '裆部', 'index': [46], 'shape_type': 'point'},
        {'name': '地面', 'index': [47, 48], 'shape_type': 'line'},
        {'name': '肚脐', 'index': [49], 'shape_type': 'point'},
        {'name': '肩凸距', 'index': [50, 51], 'shape_type': 'line'},
        {'name': '肩凹距', 'index': [52, 53], 'shape_type': 'line'},
        {'name': '肩宽', 'index': [54, 55], 'shape_type': 'line'},
        {'name': '第一上臂围', 'index': [56, 57], 'shape_type': 'line'},
        {'name': '第二上臂围', 'index': [58, 59], 'shape_type': 'line'},
        {'name': '第三上臂围', 'index': [60, 61], 'shape_type': 'line'},
        {'name': '手肘', 'index': [62, 63], 'shape_type': 'line'},
        {'name': '第一下臂围', 'index': [64, 65], 'shape_type': 'line'},
        {'name': '第二下臂围', 'index': [66, 67], 'shape_type': 'line'},
        {'name': '手腕', 'index': [68, 69], 'shape_type': 'line'},
    ]
    side_labelme_configs = [
        {'name': '侧面身高', 'index': [0, 1], 'shape_type': 'line'},
        {'name': '颈围', 'index': [2, 3], 'shape_type': 'line'},
        {'name': '腋下围', 'index': [4, 5], 'shape_type': 'line'},
        {'name': '胸围', 'index': [6, 7], 'shape_type': 'line'},
        {'name': '下胸围', 'index': [8, 9], 'shape_type': 'line'},
        {'name': '胃围', 'index': [10, 11], 'shape_type': 'line'},
        {'name': '腰围', 'index': [12, 13], 'shape_type': 'line'},
        {'name': '腹围', 'index': [14, 15], 'shape_type': 'line'},
        {'name': '臀围', 'index': [16, 17], 'shape_type': 'line'},
        {'name': '第一腿围', 'index': [18, 19], 'shape_type': 'line'},
        {'name': '第三腿围', 'index': [20, 21], 'shape_type': 'line'},
        {'name': '第五腿围', 'index': [22, 23], 'shape_type': 'line'},
        {'name': '膝上围', 'index': [24, 25], 'shape_type': 'line'},
        {'name': '膝盖围', 'index': [26, 27], 'shape_type': 'line'},
        {'name': '膝下围', 'index': [28, 29], 'shape_type': 'line'},
        {'name': '腿肚围', 'index': [30, 31], 'shape_type': 'line'},
        {'name': '脚腕围', 'index': [32, 33], 'shape_type': 'line'},
        {'name': '锁骨中', 'index': [34], 'shape_type': 'point'},
        {'name': '腿腹交界', 'index': [35], 'shape_type': 'point'},
        {'name': '腿肚下围', 'index': [36, 37], 'shape_type': 'line'},
        {'name': '脚腕以上', 'index': [38, 39], 'shape_type': 'line'},
    ]

    if data_type == 'front':
        labelme_configs = front_labelme_configs
        detector = pose_front
    elif data_type == 'side':
        labelme_configs = side_labelme_configs
        detector = pose_side
    else:
        print(f"{data_type} is error!")
        raise ValueError("error!")

    # filelist = glob.glob("D:/project/smpl/files/unlabeled/*.*")
    filelist = glob.glob(f"D:/project/smpl/files/2022.8.16/2022.8.16_{data_type}/*.*")
    for filepath in tqdm(filelist):
        basename, ext = os.path.splitext(filepath)
        if ext in [".jpg", ".jpeg", ".png"]:
            shapes = []
            img, preds = detector.do_detect(filepath)
            for labelme_config in labelme_configs:
                shape = {}
                shape['label'] = labelme_config['name']
                shape['points'] = list()
                for pts_id in labelme_config['index']:
                    shape['points'].append(preds[pts_id].tolist())
                shape['group_id'] = None
                shape['shape_type'] = labelme_config['shape_type']
                shape['flags'] = {}
                shapes.append(shape)

            with open(filepath, "rb") as f:
                imageData = base64.b64encode(f.read())

            filename = os.path.split(basename)[-1]
            imageData = imageData.decode('utf-8')
            json_data = {
                "version": "4.6.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": f"{filename}{ext}",
                "imageData": imageData,
                "imageHeight": img.shape[0],
                "imageWidth": img.shape[1],
            }
            json_path = basename + ".json"
            with open(json_path, "w") as f:
                write_data = json.dumps(json_data, indent=4)
                f.write(write_data)


