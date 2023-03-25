"""
IMPORTANT: 
This code is an adaptation of the one that we used for the project of M5 subject!
"""

import copy
import random

import cv2
import torch
from detectron2.data import build_detection_train_loader
from detectron2.engine import HookBase
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer

if torch.cuda.is_available():
    print('CUDA is available!')
else:
    print('CUDA is NOT available')

from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import os

# include the utils folder in the path
import sys
from datetime import datetime as dt

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from detectron2 import model_zoo

from src.metrics import *

from dataset import *

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# from utils.MyTrainer import *
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()  # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(
            build_detection_train_loader(self.cfg)
        )  # builds the dataloader from the provided cfg
        self.best_loss = float("inf")  # Current best loss, initially infinite
        self.weights = None  # Current best weights, initially none
        self.i = 0  # Something to use for counting the steps

    def after_step(self):  # after each step

        if self.trainer.iter >= 0:
            print(
                f"----- Iteration num. {self.trainer.iter} -----"
            )  # print the current iteration if it's divisible by 100

        data = next(self._loader)  # load the next piece of data from the dataloader

        with torch.no_grad():  # disables gradient calculation; we don't need it here because we're not training, just calculating the val loss
            loss_dict = self.trainer.model(data)  # more about it in the next section

            losses = sum(loss_dict.values())  #
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )  # puts these metrics into the storage (where detectron2 logs metrics)


                df = pd.read_csv(output_path + "/training_loss.csv")
                df = pd.concat([df, pd.DataFrame([{"total_val_loss": losses_reduced}])], axis=0, ignore_index=True)
                df.to_csv(output_path + "/training_loss.csv", index=False)

                # save best weights
                if losses_reduced < self.best_loss:  # if current loss is lower
                    self.best_loss = losses_reduced  # saving the best loss
                    self.weights = copy.deepcopy(
                        self.trainer.model.state_dict()
                    )  # saving the best weights
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        
        evaluator_list = [coco_evaluator]
        
        return DatasetEvaluators(evaluator_list)

#  https://towardsdatascience.com/train-maskrcnn-on-custom-dataset-with-detectron2-in-4-steps-5887a6aa135d

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task 1.3: Finetuning')
    parser.add_argument('--name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #
    now = dt.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    output_path = os.path.join(current_path, f'Results/Task_1.3/{dt_string}_{args.name}/{args.network}')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- DATASET --------------------------------- #
    #  Register the dataset
    classes = ['car']
    for subset in ["train", "val"]:
        DatasetCatalog.register(f"AIcity_{subset}", lambda subset=subset: get_dicts(subset, pretrained=False))
        print(f"Successfully registered 'AIcity_{subset}'!")
        MetadataCatalog.get(f"AIcity_{subset}").set(thing_classes=classes)

    metadata = MetadataCatalog.get("AIcity_train")

    # --------------------------------- MODEL ----------------------------------- #
    if args.network == 'faster_RCNN':
        model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    elif args.network == 'mask_RCNN':
        model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    else:
        print('Network not found')
        exit()

    #  Create the config
    cfg = get_cfg()

    print(cfg)

    # get the config from the model zoo
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo

    # Model
    # cfg.MODEL_MASK_ON = True  # If we want to use the mask.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    #  cfg.MODEL.BACKBONE.NAME = 'build_resnet_fpn_backbone'
    #  cfg.MODEL.BACKBONE.FREEZE_AT = 2
    #  cfg.MODEL.RESNETS.DEPTH = 50

    # Solver
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (1000, 2000, 2500)
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    # cfg.SOLVER.AMP.ENABLED = True

    # Test
    cfg.TEST.EVAL_PERIOD = 100

    # Dataset
    cfg.DATASETS.TRAIN = ("AIcity_train",)
    #  cfg.DATASETS.VAL = ("AIcity_val",)
    cfg.DATASETS.TEST = ("AIcity_val",)
    cfg.OUTPUT_DIR = output_path

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4

    print(cfg)

    # --------------------------------- TRAINING --------------------------------- #

    if "training_loss.csv" not in os.listdir(output_path):
        df = pd.DataFrame(columns=["frame_id", "total_val_loss"])
        df.to_csv(output_path + "/training_loss.csv")

    trainer = MyTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])

    # Compute the time
    start = dt.now()
    trainer.train()
    end = dt.now()
    print('Time to train: ', end - start)

    # # --------------------------------- EVALUATION --------------------------------- #
    # cfg.DATASETS.TEST = ("AIcity_val",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("AIcity_val", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "AIcity_val")

    print("-----------------Evaluation-----------------")
    print(model)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    print("--------------------------------------------")



    dataset_dicts = get_dicts('val', pretrained=True)


    bboxes_gt = {}
    bboxes_pred = {}
    for d in dataset_dicts:

        image_id = d["image_id"]

        for c in d["annotations"]:
            bbox = c["bbox"]

            if image_id not in bboxes_gt:
                bboxes_gt[image_id] = []

            bboxes_gt[image_id].append({'xtl': bbox[0],
                                        'ytl': bbox[1],
                                        'xbr': bbox[2],
                                        'ybr': bbox[3],
                                        'predicted': True,
                                        'confidence': 1.0})


        im = cv2.imread(d["file_name"])
        
        outputs = predictor(im)
        outputs_instances = outputs["instances"].to("cpu")

        bboxes = outputs_instances.pred_boxes
        for i, c in enumerate(outputs_instances.pred_classes):
            if int(c) == 0:
                bbox = bboxes[i].tensor.numpy()[0]
                
                if image_id not in bboxes_pred:
                    bboxes_pred[image_id] = []

                bboxes_pred[image_id].append({'xtl': bbox[0],
                                              'ytl': bbox[1],
                                              'xbr': bbox[2],
                                              'ybr': bbox[3],
                                              'predicted': True,
                                              'confidence': 1.0})

    rec, prec, mAP = voc_eval(bboxes_gt, bboxes_pred)

    print('mAP:', round(mAP, 4))



    # --------------------------------- INFERENCE --------------------------------- #
    dataset_dicts = get_dicts('val', pretrained=True)
    for d in random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])