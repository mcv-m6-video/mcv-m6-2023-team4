import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
import cv2

def save_video_to_img():
    file_path = "../dataset/AICity_data/train/S03/c010/vdo.avi"
    output_path = "AICity_data/S03/c010/imgs/"

    cap = cv2.VideoCapture(file_path)

    index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        print("Saving frame", index)
        cv2.imwrite(output_path + str(index) + ".png", frame)

        index += 1


    cap.release()


def parse_detection_txt(file_path):
    # Format [frame, -1, left, top, width, height, conf, -1, -1, -1].
    frames = {}
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            box_frame, _, left, top, width, height, conf, _, _, _ = map(float, row)
            box_frame = int(box_frame)
            #box_frame -= 1
            left, top, width, height, conf = map(float, [left, top, width, height, conf])
            
            if box_frame not in frames:
                frames[box_frame] = []
                
            frames[box_frame].append({
                "bbox": [left, top, width, height],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 2,
                'confidence': conf
            })
    
    return frames

def parse_xml(file_path):
    """
    Parses an XML file and extracts bounding box information for each frame and track.
    Args:
        file_path (str): Path to the XML file.
    Returns:
        tuple: A tuple containing two dictionaries, `tracks` and `frames`.
            `tracks` contains information for each track, with the track IDs as keys and the box information
            for each frame as values.
            `frames` contains information for each frame, with the frame numbers as keys and a list of boxes as
            values.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    frames = {}

    # Iterate over the tracks and extract their bounding box information
    for track in root.findall(".//track[@label='car']"):

        for box in track.findall(".//box"):
            box_frame = int(box.get('frame'))
            xtl, ytl, xbr, ybr = map(float, [box.get('xtl'), box.get('ytl'), box.get('xbr'), box.get('ybr')])
            outside, occluded, keyframe = map(int, [box.get('outside'), box.get('occluded'), box.get('keyframe')])
            parked = box.find(".//attribute[@name='parked']").text == 'true'

            # Add the box to the list of boxes for this frame
            if box_frame not in frames:
                frames[box_frame] = []

            frames[box_frame].append({
                "bbox": [xtl, ytl, xbr, ybr],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                'outside': outside,
                'occluded': occluded,
                'keyframe': keyframe,
                'parked': parked,
                'predicted': False,
                'detected': True,
                'confidence': float(100)
            })

    return frames


def get_dicts(subset, pretrained = False):
    images = "AICity_data/S03/c010/imgs"

    sequences_id = os.listdir(images)

    sequences_id = list(sorted(sequences_id, key = lambda s: int(str(s)[0:-4])))

    if subset == "all":
        sequences_id = sequences_id
        
    elif subset == "train":
        sequences_id = sequences_id[0:535]

    elif subset == "val":
        sequences_id = sequences_id[535:]

    dataset_dicts = []

    bboxes_gt = parse_xml("AICity_data/S03/c010/ai_challenge_s03_c010-full_annotation.xml") #parse_detection_txt("AICity_data/S03/c010/gt.txt")

    for seq_id in tqdm(sequences_id):
            
        record = {}

        filename = os.path.join(images, str(seq_id))

        height, width = cv2.imread(filename).shape[:2]

        idx = int(str(seq_id)[0:-4])

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        if idx in bboxes_gt:
            record["annotations"] = bboxes_gt[idx]
        else:
            record["annotations"] = {}

        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    save_video_to_img()