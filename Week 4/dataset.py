import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer
import sys
import os
from pathlib import Path
import glob
import json


class Dataset:
    def __init__(self, train_sequences=['S01', 'S04'], test_sequences=['S03']):
        for sequence in train_sequences+test_sequences:
            self.save_sequence_to_frames(sequence)
    
    def save_sequence_to_frames(self, sequence='S03'):
        for cam_path in glob.glob('../datasets/AITrack/train/'+sequence+'/*'):
            output_path = './outputs/frames/'+sequence+'/'+os.path.basename(cam_path)+'/' 
            if not os.path.exists(output_path):
                print('Saving video to frames of sequence-camera: %s-%s' %(sequence, os.path.basename(cam_path)))
                self.save_video_to_frames(cam_path+'/vdo.avi', output_path='./outputs/frames/'+sequence+'/'+os.path.basename(cam_path)+'/')
    
    def save_video_to_frames(self, file_path='../dataset/AITrack/train/S03/c010/vdo.avi', output_path='./outputs/frames/S03/c010/'):
        Path(output_path).mkdir(parents=True, exist_ok=True) 
        
        cap = cv2.VideoCapture(file_path)
        index = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            cv2.imwrite(output_path + str(index) + ".png", frame)
            index += 1

        cap.release()

    # TODO - Implement option to only take into account regions of interest
    # Maybe put frames in black (constant value)?
    def get_dicts_from_sequences(self, sequences=['S01', 'S04'], only_roi=False):
        dataset_dicts = []
        sequences.sort()
        sequences_dicts_path = './outputs/frames/'+'_'.join(sequences)
        
        if only_roi:
            sequences_dicts_path + '_only_roi'
        
        sequences_dicts_path += '_json'
        
        if os.path.isfile(sequences_dicts_path):
            with open(sequences_dicts_path, 'r') as fout:
                dataset_dicts = json.load(fout)
                return dataset_dicts
        
        for sequence in sequences:
            for cam_path in glob.glob('../datasets/AITrack/train/'+sequence+'/*'):
                cam = os.path.basename(cam_path)
                bboxes_gt = self.parse_txt('../datasets/AITrack/train/'+sequence+'/'+cam+'/gt/gt.txt')
                
                for frame_index, frame_path in enumerate(glob.glob(f'./outputs/frames/{sequence}/{cam}/*.png')):
                    record = {}

                    height, width = cv2.imread(frame_path).shape[:2]

                    record["file_name"] = frame_path
                    record["image_id"] = sequence+'_'+cam+'_'+str(frame_index)
                    record["height"] = height
                    record["width"] = width

                    if frame_index in bboxes_gt:
                        record["annotations"] = bboxes_gt[frame_index]
                    else:
                        record["annotations"] = {}

                    dataset_dicts.append(record)
        
        with open(sequences_dicts_path, 'w') as fout:
            json.dump(dataset_dicts, fout)
       
        return dataset_dicts

    def parse_txt(self, file_path):
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
                    "category_id": 0,
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

def get_splitted_sequences(type='Normal', k=4):
    images = "AICity_data/S03/c010/imgs"

    sequences_id = os.listdir(images)

    sequences_id = list(sorted(sequences_id, key = lambda s: int(str(s)[0:-4])))
    
    sequences_id = np.array(sequences_id)
    
    if type == 'Random':
        # It modifies the argument in place
        np.random.shuffle(sequences_id)
    
    splitted_sequences = np.array_split(np.array(sequences_id), 4)
    
    
    return splitted_sequences

def get_dicts_from_sequences(sequences_id):
    images = "AICity_data/S03/c010/imgs"

    dataset_dicts = []

    bboxes_gt = parse_xml("AICity_data/S03/c010/ai_challenge_s03_c010-full_annotation.xml")

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