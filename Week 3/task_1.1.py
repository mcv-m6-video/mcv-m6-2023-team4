import random

import cv2
from PIL import Image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()

# import some common libraries
import argparse

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

# import some common detectron2 utilities
from detectron2 import model_zoo

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pycocotools.mask as mask_utils

from src.metrics import *

from dataset import *

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task 1.1')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    parser.add_argument('--th', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    classes =  {0: u'car'}

    for subset in ["all", "train", "val"]:
        DatasetCatalog.register(f"AIcity_{subset}", lambda subset=subset: get_dicts(subset, pretrained=True))
        MetadataCatalog.get(f"AIcity_{subset}").set(thing_classes=list(classes.values()))

    # Config
    cfg = get_cfg()

    output_path = 'Results/Task_1.1/' + args.network + '/' + str(args.th) + '/'

    if args.network == 'faster_RCNN':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th  # set threshold for this model
    cfg.DATASETS.TEST = ("AIcity_all",)

    
    # Predictor
    predictor = DefaultPredictor(cfg)

    # Evaluator
    evaluator = COCOEvaluator("AIcity_all", cfg, False, output_dir=output_path)

    # Evaluate the model
    """
    loader = build_detection_test_loader(cfg, "AIcity_all")
    print(inference_on_dataset(predictor.model, loader, evaluator))
    """

    dataset_dicts = get_dicts('all', pretrained=True)


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
            if int(c) == 2:
                bbox = bboxes[i].tensor.numpy()[0]
                
                if image_id not in bboxes_pred:
                    bboxes_pred[image_id] = []

                bboxes_pred[image_id].append({'xtl': bbox[0],
                                              'ytl': bbox[1],
                                              'xbr': bbox[2],
                                              'ybr': bbox[3],
                                              'predicted': True,
                                              'confidence': 1.0})

        print("Evaluated image", image_id)

    rec, prec, mAP = voc_eval(bboxes_gt, bboxes_pred)

    print('mAP:', round(mAP, 4))



    # Save the images and gif with bboxes:

    video_frames_out = []    
    for d in dataset_dicts[1500:1800]: #random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        outputs_instances = outputs["instances"].to("cpu")
        
        #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        #out = v.draw_instance_predictions(outputs_instances)
        #cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        fig, ax = plt.subplots(1, 1, figsize=(10,8), dpi=100)
        ax.set_title("Frame: " + str(d["image_id"]) + "\nNetwork: " + str(args.network) + "\nThreshold: " + str(args.th))
        ax.imshow(im[:, :, ::-1])
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        for c in d["annotations"]:
            bbox = c["bbox"]

            ax.add_patch(patches.Rectangle((bbox[0], bbox[1]),
                                            bbox[2] - bbox[0],
                                            bbox[3] - bbox[1],
                                            linewidth=2, edgecolor='blue', facecolor='none'))

        bboxes = outputs_instances.pred_boxes
        for i, c in enumerate(outputs_instances.pred_classes):
            if int(c) == 2:
                bbox = bboxes[i].tensor.numpy()[0]

                ax.add_patch(patches.Rectangle((bbox[0], bbox[1]),
                                                bbox[2] - bbox[0],
                                                bbox[3] - bbox[1],
                                                linewidth=2, edgecolor='red', facecolor='none'))
                

        ax.legend(handles=[patches.Patch(color='red', label='Prediction'), patches.Patch(color='blue', label='Groundtruth')])
                
        fig.savefig(output_path + d["file_name"].split('/')[-1])

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        frame_out = Image.fromarray(np.asarray(canvas.buffer_rgba()).astype("uint8"))
        video_frames_out.append(frame_out)
        canvas.get_renderer().clear()
        plt.close(fig)

        print("Processed image: " + d["file_name"].split('/')[-1])

    video_frames_out[0].save(output_path + 'Output_' + 'mAP=' + str(round(mAP, 4)) + '.gif', save_all=True, append_images=video_frames_out[1:], duration=30, loop=0)