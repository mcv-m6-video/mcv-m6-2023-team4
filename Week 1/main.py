import sys
import xml.etree.ElementTree as ET
import cv2
import os
import os
import cv2
import subprocess
from IPython.display import Video, display
import numpy as np
import copy

def parse_gt_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # split the line into columns using a comma as delimiter
            columns = line.strip().split(',')
            
            # extract the first six columns and join them into a string
            output = ','.join(columns[:6]).split(',')
            
            # print the output string
            print(output)
            data.append(output)
            
                      
# parse_gt_txt('./mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/gt/gt.txt')

def parse_xml(file_path):
    # parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    tracks = {}
    frames = {}

    # iterate over the frames
    for track in root.findall('.//track'):
        track_id = track.get('id')
        label = track.get('label')
        # print('Track id:', track_id)
        
        if label != 'car':
            continue
        
        tracks[track_id] = {}
        
        # iterate over the boxes in the track and extract their attributes
        for box in track.findall('.//box'):
            box_frame = box.get('frame')
            box_frame = int(box_frame)
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            outside = int(box.get('outside'))
            occluded = int(box.get('occluded'))
            keyframe = int(box.get('keyframe'))
            parked = box.find(".//attribute[@name='parked']")
            
            if parked.text == 'true':
                parked = True
            else:
                parked = False
                
            tracks[track_id][box_frame] = {
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr,
                'outside': outside,
                'occluded': occluded,
                'keyframe': keyframe,
                'parked': parked,
                'noisy': False
            }
            
            # Each frame has a list of bounding boxes
            if box_frame not in frames:
                frames[box_frame] = []
                
            frames[box_frame].append(
                {
                    'track_id': int(track_id),
                    'xtl': xtl,
                    'ytl': ytl,
                    'xbr': xbr,
                    'ybr': ybr,
                    'outside': outside,
                    'occluded': occluded,
                    'keyframe': keyframe,
                    'parked': parked,
                    'noisy': False
                }
            )
                
            
            # print('Box:', box_frame, 'xtl:', xtl, 'ytl:', ytl, 'xbr:', xbr, 'ybr:', ybr, 'outside:', outside, 'occluded:', occluded, 'keyframe:', keyframe, 'parked:', parked)
    
    return tracks, frames
        
  
def parse_video(file_path, frames):
    # load the video file
    cap = cv2.VideoCapture(file_path)
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    index = 0
    # loop through the frames
    while True:
        # read a frame from the video
        ret, frame = cap.read()
        

        # check if the frame was successfully read
        if not ret:
            break
        
        mean_iou, frames_with_iou = calculate_mean_iou(frames[index])
        mean_ap = calculate_mean_ap(frames[index])
                
        num_frames_normal = 0
        num_frames_noisy = 0
        for bounding_box in frames[index]:
            # convert coordinates to integers
            xtl, ytl, xbr, ybr = map(int, [bounding_box['xtl'], bounding_box['ytl'], bounding_box['xbr'], bounding_box['ybr']])
            
            print(bounding_box)
            # draw rectangle on image
            if bounding_box['noisy']:
                num_frames_noisy += 1
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
            elif bounding_box['parked']:
                num_frames_normal += 1
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
            else:
                num_frames_normal += 1
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
        
        print('Normal:', num_frames_normal)
        print('Noisy:', num_frames_noisy)
        # for bounding_box in frame
        # display the frame
        cv2.imshow('frame', frame)
        
        # wait for a key press
        # key = cv2.waitKey(1000)
        
        key = cv2.waitKey(250000000)

        # check if the user pressed the 'q' key to quit
        if key == ord('q'):
            break
    
        index += 1

    # release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()

tracks, frames = parse_xml('mcv-m6-2023-team4/datasets/ai_challenge_s03_c010-full_annotation.xml')
# print(tracks['9']['86'])
# print(frames[86])
# print(frames[0])

# parse_video('mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/vdo.avi', frames)

def create_random_bbox(img_width, img_height, min_size, max_width, max_height):
    dp = np.random.randint(0,300)
    x1 = max(10, 384*np.random.random())
    y1 = max(10, 384*np.random.random())
    x2 =  np.random.randint(0, min(x1+dp, 1920-x1))
    y2 =  np.random.randint(0, min(y1+dp,1920-y1))
        
    # if x1 >= img_width or y2 >= img_height:
    #     create_random_bbox(img_width, img_height, max_width, max_height)
    
    return x1, y1, x2, y2


def bbox_iou(bboxA, bboxB):
    # Code provided by teacher in M1
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

def calculate_mean_iou(frame):
    bboxes_noisy = []
    bboxes_normal = []
    
    
    # iterate over each dictionary in the array
    for d in frame:
        if d['noisy']:
            # if noisy is True, append to the noisy_true list
            d['iou'] = 0
            d['detected'] = False
            bboxes_noisy.append(d)
        else:
            # if noisy is False, append to the noisy_false list
            bboxes_normal.append(d)
    
    #generate mean iou
    mean_iou = 0
    used_indexes_noisy = []
    iou_bboxes_normal = []
    
    for bbox_normal in bboxes_normal:
        bbox_coords_norm = [bbox_normal['xtl'], bbox_normal['ytl'], bbox_normal['xbr'], bbox_normal['ybr']]
        max_iou = 0
        max_index = -1
        for index, bbox_noisy in enumerate(bboxes_noisy):
            if index not in used_indexes_noisy:
                bbox_coords_noisy = [bbox_noisy['xtl'], bbox_noisy['ytl'], bbox_noisy['xbr'], bbox_noisy['ybr']]
                iou = bbox_iou(bbox_coords_norm, bbox_coords_noisy)
                if iou > max_iou:
                    max_iou = iou
                    max_index = index
        
        # Each GT box can only be assigned to one predicted box
        iou_bboxes_normal.append(max_iou)
        used_indexes_noisy.append(max_index)
        
        bbox_normal['iou'] = max_iou
        bboxes_noisy[max_index]['detected'] = True
        bboxes_noisy[max_index]['iou'] = max_iou
        
    
    mean_iou = np.array(iou_bboxes_normal).mean()
    
    return mean_iou, bboxes_normal, bboxes_noisy

def generate_confidence(frame):
    for bbox in frame:
        bbox['confidence'] = np.random.uniform(0,1)
    return frame

def order_frame_by_confidence(frame):
    return sorted(frame, key=lambda x: x['confidence'], reverse=True)

def calculate_mean_ap(frame_groundtruth, frame_preds):
    N = 10
    iou_thresh = 0.5
    
    mean_ap = 0
    
    for i in range(N):
        # We generate confidence scores and sort    
        frame_preds = generate_confidence(frame_preds)
        sorted_frame_preds = order_frame_by_confidence(frame_preds)
        
        # Compute the precision and recall values for different decision thresholds
        thresholds = [bbox['confidence'] for bbox in frame_preds]
        
        precisions = []
        recalls = []
        
        detected = 0
        for threshold in thresholds:
            tp = 0
            fp = 0

            for bbox in sorted_frame_preds:
                if bbox['detected'] == True:
                    detected += 1
                if bbox['confidence'] > threshold:
                    iou = bbox['iou']
                    if iou > iou_thresh:
                        tp += 1
                    else:
                        fp += 1
     
            precisions.append(tp/(tp+fp))
            recalls.append(tp/len(frame_groundtruth))
        
        
        # If GT object does not have any prediction we consider that we try to find the object with an infinite amount of bounding boxes           
         # precision = (tp / [tp + fp]) = (1 / inf) = 0 -> lots of false positive because of the infinite amount of bounding boxes
         # recall = (tp / [tp + fn]) = (1 / 1) = 1 -> fn = 0 because of no object not detected

         #Therefore, if we have not detected all the bounding boxes in the ground truth we append
        if detected < len(frame_groundtruth):
            precisions.append(0)
            recalls.append(1)
            
        mean_ap += map_pascal_VOC(precisions, recalls)

    return mean_ap     
        

        
    
    
    # results_per_category = []
    # for idx, name in enumerate(class_names):
    #     # area range index 0: all area ranges
    #     # max dets index -1: typically 100 per image
    #     precision = precisions[:, :, idx, 0, -1]
    #     precision = precision[precision > -1]
    #     ap = np.mean(precision) if precision.size else float("nan")
    #     results_per_category.append(("{}".format(name), float(ap * 100)))
    
def map_pascal_VOC(precisions, recalls):
    index_recalls = len(recalls) - 2
    index_precisions = len(recalls) - 1
    average_precision = 0 
    for i in range(1, -0.1, -0.1):
        if i < recalls[index_recalls]:
            if index_recalls != 0:
                index_recalls -= 1
                index_precisions -= 1
            elif index_recalls == 0 and index_precisions != 0:
                index_precisions -= 1

        average_precision += precisions[index_precisions]

    return average_precision / 11



def generate_noisy_annotations(frames, th_dropout, th_generate, mean, std):
    frames_all = copy.deepcopy(frames)
    frames_noisy = {}
    
    for index, frame in frames.items():                  
        frames_noisy[index] = []
        for bounding_box in frame:
            dropout = np.random.uniform(0,1)
            
            if dropout > th_dropout:
                # Move box / make it bigger
                dx1, dy1, dx2, dy2 = mean + std * np.random.randn(4)
                frame_noisy = {
                        'track_id': bounding_box['track_id'],
                        'xtl': bounding_box['xtl'] + dx1,
                        'ytl': bounding_box['ytl'] + dy1,
                        'xbr': bounding_box['xbr'] + dx2,
                        'ybr': bounding_box['ybr'] + dy2,
                        'outside': None,
                        'occluded': None,
                        'keyframe': None,
                        'parked': None,
                        'noisy': True,
                    }
                
                frames_noisy[index].append(frame_noisy)
                frames_all[index].append(frame_noisy)
            
        # In each frame we generate or not a not annotated bounding box
        generate = np.random.uniform(0,1)
        
        if generate > th_generate:
            bounding_boxes_gen_num = np.random.randint(1, 4)
            for i in range(bounding_boxes_gen_num):
                xtl, ytl, xbr, ybr = create_random_bbox(1920,1080, 40, np.random.randint(250, 400),np.random.randint(250,400))
                frame_noisy = {
                        'track_id': None,
                        'xtl': xtl,
                        'ytl': ytl,
                        'xbr': xbr,
                        'ybr': ybr,
                        'outside': None,
                        'occluded': None,
                        'keyframe': None,
                        'parked': None,
                        'noisy': True
                    }
                
                frames_noisy[index].append(frame_noisy)
                frames_all[index].append(frame_noisy)
        
    return frames_noisy, frames_all

# sys.exit()

# bboxA = {}
# bboxB = {}
# print(bbox_iou())

frames_noisy, frames_all = generate_noisy_annotations(frames, th_dropout=0.3, th_generate=0.3, mean=0, std=10)

parse_video('mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/vdo.avi', frames_all)