"""
IMPORTANT: 
This code is an adaptation of the one that we used for the project of M5 subject!
"""

import copy
import random
import distutils.core
import cv2
import torch
import sys
import os


dist = distutils.core.run_setup("../detectron2/setup.py")
sys.path.insert(0, os.path.abspath('../detectron2'))

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


# include the utils folder in the path

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

np.random.seed(42)

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

#  https://towardsdatascience.com/train-maskrcnn-on-custom-dataset-with-detectron2-in-4-steps-5887a6aa135d

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task 1.4: Cross-validation')
    parser.add_argument('--name', type=str, default='baseline', help='Name of the experiment')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--cross_validation', type=str, default='None', help='Use cross validation or not: None, Normal or Random')
    parser.add_argument('--index', type=str, default='None', help='Index where to start cross validation, it can be a number between 0 and 3')
    parser.add_argument('--only_inference', type=str, default='False', help='If no training has to be done: False or True')
    parser.add_argument('--th', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    train_sequences = ['S01']
    test_sequences = ['S04']
    dataset = Dataset(train_sequences, test_sequences)
    
    mAPs = []
    if args.cross_validation == 'Normal' or args.cross_validation == 'Random':
        k = 4
        splitted_sequences = get_splitted_sequences(args.cross_validation, k=k)
    elif args.cross_validation == 'None':
        k = 1
    
    for index in range(k):
        # --------------------------------- OUTPUT --------------------------------- #
        now = dt.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

        # TODO - Put a good path to save
        output_path = os.path.join(current_path, f'Results/Task_1.4/{args.cross_validation}_{str(index)}_{dt_string}_{args.name}/{args.network}')

        os.makedirs(output_path, exist_ok=True)

        # --------------------------------- DATASET --------------------------------- #
        #  Register the dataset
        DatasetCatalog.clear()
        classes = ['car']
        if args.cross_validation == 'None':
            subset = 'train'
            DatasetCatalog.register(f"AIcity_{subset}", lambda subset=subset: dataset.get_dicts_from_sequences(sequences=train_sequences))
            print(f"Successfully registered 'AIcity_{subset}'!")
            MetadataCatalog.get(f"AIcity_{subset}").set(thing_classes=classes)
            
            subset = 'val'
            DatasetCatalog.register(f"AIcity_{subset}", lambda subset=subset: dataset.get_dicts_from_sequences(sequences=test_sequences))
            print(f"Successfully registered 'AIcity_{subset}'!")
            MetadataCatalog.get(f"AIcity_{subset}").set(thing_classes=classes)
            
            metadata = MetadataCatalog.get("AIcity_train")
            
            dataset_catalog_train = "AIcity_train"
            dataset_catalog_valid = "AIcity_val"
        # TODO - Remove this or put this to use cross-val, this is not used anymore
        elif args.cross_validation == 'Normal' or args.cross_validation == 'Random' :
            train_sequence = splitted_sequences[index]
            val_sequence = splitted_sequences[:index] + splitted_sequences[index+1:]
            val_sequence = [item for sublist in val_sequence for item in sublist]
            
            dataset_catalog_train = f"AIcity_{args.cross_validation}_{str(index)}_{str(args.lr)}_train"
            dataset_catalog_valid = f"AIcity_{args.cross_validation}_{str(index)}_{str(args.lr)}_val"
        
            DatasetCatalog.register(dataset_catalog_train, lambda subset="train": get_dicts_from_sequences(train_sequence))
            print(f"Successfully registered 'AIcity_train'!")
            MetadataCatalog.get(dataset_catalog_train).set(thing_classes=classes)
            
            DatasetCatalog.register(dataset_catalog_valid, lambda subset="val": get_dicts_from_sequences(val_sequence))
            print(f"Successfully registered 'AIcity_val'!")
            MetadataCatalog.get(dataset_catalog_valid).set(thing_classes=classes)

            metadata = MetadataCatalog.get(dataset_catalog_train)

        # --------------------------------- MODEL ----------------------------------- #
        # TODO - Test RetinaNet and add DETR (https://github.com/facebookresearch/detr/tree/main/d2)
        if args.network == 'faster_RCNN':
            model = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        elif args.network == 'retinanet':
            model = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
        elif args.network == 'DETR':
            model = ''
        elif args.network == 'mask_RCNN':
            model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
        else:
            print('Neural network not found')
            exit()

        #  Create the config
        cfg = get_cfg()

        # get the config from the model zoo
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo

        # Model
        # cfg.MODEL_MASK_ON = True  # If we want to use the mask.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1  # faster, and good enough for this toy dataset (default: 512)
        #  cfg.MODEL.BACKBONE.NAME = 'build_resnet_fpn_backbone'
        #  cfg.MODEL.BACKBONE.FREEZE_AT = 2
        #  cfg.MODEL.RESNETS.DEPTH = 50

        # Solver
        cfg.SOLVER.BASE_LR = args.lr
        # Before 3000
        # cfg.SOLVER.MAX_ITER = 1
        cfg.SOLVER.MAX_ITER = args.epochs
        cfg.SOLVER.STEPS = (1000, 2000, 2500)
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.CHECKPOINT_PERIOD = 100
        # cfg.SOLVER.AMP.ENABLED = True

        # Test
        cfg.TEST.EVAL_PERIOD = 100

        # Dataset
        cfg.DATASETS.TRAIN = (dataset_catalog_train,)
        #  cfg.DATASETS.VAL = ("AIcity_val",)
        cfg.DATASETS.TEST = (dataset_catalog_valid,)
        cfg.OUTPUT_DIR = output_path

        # Dataloader
        cfg.DATALOADER.NUM_WORKERS = 1
        
        # --------------------------------- TRAINING --------------------------------- #
        if args.only_inference == 'False':
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
            
            sys.exit()
        

        # # # --------------------------------- EVALUATION --------------------------------- #
        # cfg.DATASETS.TEST = ("AIcity_val",)
        if args.only_inference == 'False':
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        else:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th 
            if args.network == 'faster_RCNN':
                cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
            elif args.network == 'mask_RCNN':
                cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

        predictor = DefaultPredictor(cfg)

        evaluator = COCOEvaluator(dataset_catalog_valid, cfg, False, output_dir=output_path)
        val_loader = build_detection_test_loader(cfg, dataset_catalog_valid)

        print("-----------------Evaluation-----------------")
        print(model)
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
        print("--------------------------------------------")


        if args.cross_validation == 'None':
            dataset_dicts = get_dicts('val', pretrained=True)
        else:
            dataset_dicts = get_dicts_from_sequences(val_sequence)


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
        mAPs.append(mAP)



        # --------------------------------- INFERENCE --------------------------------- #
        if args.only_inference == 'False':
            if args.cross_validation == 'None':
                dataset_dicts = get_dicts('val', pretrained=True)
            else:
                dataset_dicts = get_dicts_from_sequences(val_sequence)
                
            for d in random.sample(dataset_dicts, 30):
                im = cv2.imread(d["file_name"])
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

                print("Processed image: " + d["file_name"].split('/')[-1])
            
    print(mAPs)