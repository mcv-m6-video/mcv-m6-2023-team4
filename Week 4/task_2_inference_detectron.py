import copy
import random
import distutils.core
import cv2
import torch
import sys
import os
from PIL import Image
import numpy as np
import shutil
from video import Video as Video
import glob

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

with open('trackeval_results.npy', 'rb') as f:
    cams = np.load(f)
    hotas = np.load(f)
    idfs = np.load(f)
    
print(cams)
print(hotas)
print(idfs)

def compute_iou(bboxA, bboxB):
    # Code provided by teacher in M1 subject
    # compute the intersection over union of two bboxes
    
    # Format of the bboxes is [xtl, ytl, xbr, ybr, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[3] - bboxA[1] + 1) * (bboxA[2] - bboxA[0] + 1)
    bboxBArea = (bboxB[3] - bboxB[1] + 1) * (bboxB[2] - bboxB[0] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou

def save_detection_txt(detection_dict, output_file):
    # Open the output file for writing
    output_fp = open(output_file, "w")

    for frame in detection_dict:
        for bbox in detection_dict[frame]:
            x, y, z = -1, -1, -1  # No information about x, y, z
            line = "{},{},{},{},{},{},{},{},{},{}\n".format(
                str(int(frame)+1),
                bbox['id'],
                bbox['xtl'],
                bbox['ytl'],
                bbox['xbr'] - bbox['xtl'],
                bbox['ybr'] - bbox['ytl'],
                1,
                x,
                y,
                z
            )

            output_fp.write(line)

    #Release the output file
    output_fp.close()

def trackeval_video(sequence='S03', cam='c010'):
    root_video_path = f'../datasets/AITrack/train/{sequence}/{cam}/'

    shutil.copy(f'../datasets/AITrack/train/{sequence}/{cam}/gt/gt.txt',
                "../../../TrackEval/data/gt/mot_challenge/Faster-RCNN-finetuned-train/Faster-RCNN-finetuned-03/gt/gt.txt")

    # Load the video
    video = cv2.VideoCapture(root_video_path+'vdo.avi')
    video_object = Video(root_video_path+'vdo.avi')


    #Load configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = './outputs/models/faster_rcnn_finetuned_3000.pth'
    predictor = DefaultPredictor(cfg)


    # Initialize variables

    object_tracker = {}
    current_id = 0
    frame_num = 1

    visualize = False

    #Initialize video array
    video_output = []
    # Open the output file for writing
    output_file = "../TrackEval/data/trackers/mot_challenge/Faster-RCNN-finetuned-train/S03/data/Faster-RCNN-finetuned-03.txt"
    
    output_fp = open(output_file, "w")
    while True:
        # Read the current frame from the video
        
        print(f'Frame num: {frame_num}')
        ret, frame = video.read()

        # Stop if there are no more frames
        if not ret:
            break

        # Detect objects in the current frame using the pretrained model
        outputs = predictor(frame) 
        current_objects = outputs['instances']

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        # print("Boxes", boxes)
        classes = instances.pred_classes.numpy()
        # print("Classes", classes)
        #print(classes)
        scores = instances.scores.numpy()
        num_boxes = boxes.shape[0]
        # Initialize a new dictionary to store the detected objects in the current frame
        new_object_tracker = {}

        for i in range(num_boxes):
            box = boxes[i]
            score = scores[i]
            class_object = classes[i]
            # Filter out low-scoring objects
            if score < 0.5:
                continue

            # if class_object != 2:
            #     continue

            # Assign a new ID to each new detected object
            current_id += 1
            # Try to match the detected object with a previously tracked object based on IoU
            best_match_id = None
            best_match_iou = 0
            for object_id, object_box in object_tracker.items():
                iou = compute_iou(box, object_box)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_id = object_id
            # If the best match has IoU > 0.4, assign the same ID to the detected object
            if best_match_id is not None and best_match_iou > 0.4:
                new_object_tracker[best_match_id] = box
                del object_tracker[best_match_id]
            else:
                new_object_tracker[current_id] = box

        if num_boxes > 0:
            # Update the object tracker for the current frame
            object_tracker = new_object_tracker
            
            if visualize:
                # Visualize the tracked objects in the current frame
                tracked_boxes = []
                tracked_ids = []
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                index_color = 0
                for object_id, object_box in object_tracker.items():
                    tracked_boxes.append(object_box)
                    tracked_ids.append(object_id)
                    out = v.draw_text(f"{object_id}", (object_box[0], object_box[1]), font_size=8)
                    out = v.draw_box(object_box, )
                
                result = out.get_image()[:, :, ::-1]
        else:
            if visualize:
                result = frame
                
        if visualize:
            video_output.append(Image.fromarray(result))
        # # Display the current frame with the tracked objects
        # cv2.imshow("Object tracking", result)
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break

        # Write the tracker output to the output file in the MOT16 format
        for track, bbox in object_tracker.items():
            x, y, z = -1, -1, -1  # No information about x, y, z
            line = "{},{},{},{},{},{},{},{},{},{}\n".format(
                frame_num,
                track,
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                1,
                x,
                y,
                z
            )
            output_fp.write(line)

        frame_num += 1




    # Release the video and the output file and close all windows
    video.release()
    # cv2.destroyAllWindows()
    output_fp.close()
    
    
    # Remove incorrect bounding boxes
    predictions = video_object.parse_detection_txt(output_file, roi_path=root_video_path+'roi.jpg', remove_parked_cars=True)
    save_detection_txt(predictions, output_file)

    
    # Modify seqinfo.ini file for trackeval
    ini_path = '../TrackEval/data/gt/mot_challenge/Faster-RCNN-finetuned-train/Faster-RCNN-finetuned-03/seqinfo.ini'
    f = open(ini_path, "r")
    output = f.read()
    f.close()
    output = output.splitlines()
    print(output)
    output[4] = f'seqLength={frame_num}'
    print(output)
    f = open(ini_path, "w")
    f.write("\n".join(output))
    f.close()
    
    output = os.popen("python ../TrackEval/scripts/run_mot_challenge.py  --BENCHMARK Faster-RCNN-finetuned --DO_PREPROC False --OUTPUT_DETAILED False\
     --OUTPUT_SUMMARY False --TIME_PROGRESS False --OUTPUT_EMPTY_CLASSES False --PLOT_CURVES False --PRINT_CONFIG False").read()


    print('Output')
    print(output)

    output = output.splitlines()
    print(output)
    for index, elem in enumerate(output):
        print(f'{index}: {elem}')

    hota = float(output[12][35:40])
    idf1 = float(output[20][35:40])

    print(f'{hota}-{idf1}')

    
    
    
    return frame_num, hota, idf1


hotas = []
idfs = []
cams = []

max_index = np.Inf
index = 0
for sequence_path in glob.glob('../datasets/AITrack/train/*'):
    if index > max_index:
        break
    sequence = os.path.basename(sequence_path)
    for cam_path in glob.glob(sequence_path + '/*'):
        cam = os.path.basename(cam_path)
        cams.append(cam)

        # frame_num, hota, idf1 = trackeval_video(sequence, cam)
        hota = 0
        idf1 = 0
        
        hotas.append(hota)
        idfs.append(idf1)

        if index > max_index:
            break
        index += 1
    
print(cams)
print(hotas)
print(idfs)

with open('trackeval_results.npy', 'wb') as f:
    np.save(f, np.array(cams))
    np.save(f, np.array(hotas))
    np.save(f, np.array(idfs))
    
