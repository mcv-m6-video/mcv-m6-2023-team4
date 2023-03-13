
import xml.etree.ElementTree as ET
import cv2
from src.utils import *
from src.metrics import *
import csv
import sys
import matplotlib.pyplot as plt


#TODO
def parse_detection_txt(file_path):
    # Format [frame, -1, left, top, width, height, conf, -1, -1, -1].
    frames = {}
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            box_frame, _, left, top, width, height, conf, _, _, _ = map(float, row)
            box_frame = int(box_frame)
            box_frame -= 1
            left, top, width, height, conf = map(float, [left, top, width, height, conf])
            
            if box_frame not in frames:
                frames[box_frame] = []
                
            frames[box_frame].append({
                'xtl': left,
                'ytl': top,
                'xbr': left + width,
                'ybr': top + height,
                'outside': None,
                'occluded': None,
                'keyframe': None,
                'parked': None,
                'predicted': True,
                'detected': False,
                'confidence': conf
            })
    
    return frames    
    
# frames = parse_detection_txt('C:/Users/Marcos/Desktop/Master Computer Vision/M6/Project/Work/mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')

# parse_gt_txt('./mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/gt/gt.txt')

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
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr,
                'outside': outside,
                'occluded': occluded,
                'keyframe': keyframe,
                'parked': parked,
                'predicted': False,
                'detected': False,
                'confidence': float(0)
            })

    return frames


def parse_video(file_path, frames, show_video=False, without_confidence=False):
    # load the video file
    cap = cv2.VideoCapture(file_path)
    
    if show_video == True:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    mean_ap = 0
    video_iou = []
    
    index = 0
    # loop through the frames
    while True:
        # read a frame from the video
        ret, frame = cap.read()
        

        # check if the frame was successfully read
        if not ret:
            break
        
        
        mean_iou, frames_with_iou, bboxes_noisy = calculate_mean_iou(frames[index])
        ap = calculate_ap(frames[index], bboxes_noisy, without_confidence)
        mean_ap += ap
        video_iou.append(mean_iou)
        
        for bounding_box in frames[index]:
            draw_rectangle_on_frame(frame, bounding_box)
        
        print('Mean iou:', mean_iou)
        
        # display the frame
        if show_video == True:
            cv2.imshow('frame', frame)
            # wait for a key press     
            key = cv2.waitKey(250000000)

            # check if the user pressed the 'q' key to quit
            if key == ord('q'):
                break
    
        index += 1

    mean_ap = mean_ap /index
    print('Mean ap:', mean_ap)
    
    
    # release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    plt.plot(list(range(index)), video_iou)
    plt.ylim(0,1)
    plt.show()

