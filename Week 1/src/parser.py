
import xml.etree.ElementTree as ET
import cv2
from src.utils import *
from src.metrics import *


#TODO
def parse_gt_txt(file_path):
    pass

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
        
        mean_iou, frames_with_iou, bboxes_noisy = calculate_mean_iou(frames[index])
        mean_ap = calculate_mean_ap(frames[index], bboxes_noisy)
                
        for bounding_box in frames[index]:
            draw_rectangle_on_frame(frame, bounding_box)
        
        # display the frame
        cv2.imshow('frame', frame)
        
        # wait for a key press     
        key = cv2.waitKey(250000000)

        # check if the user pressed the 'q' key to quit
        if key == ord('q'):
            break
    
        index += 1

    # release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()

