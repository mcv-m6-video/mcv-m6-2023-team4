
import xml.etree.ElementTree as ET
import cv2
from src.utils import *
from src.metrics import *
import csv
import sys
import matplotlib.pyplot as plt
import threading
import time
import matplotlib.animation as animation
from functools import partial
import os
import subprocess
import glob
from moviepy.editor import VideoFileClip

# root_folder = 'C:/Users/Marcos/Desktop/Master Computer Vision/M6/Project/Work/test/mcv-m6-2023-team4/Week 1/'
root_folder = '../'

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

    
def parse_video(file_path, frames, show_video=False, without_confidence=False,generate_plot_video=False):
    # load the video file
    cap = cv2.VideoCapture(file_path)
    
    
    if show_video == True:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    
    start_plot = 0
    end_plot = 100
    # end_plot = np.Inf
    
    mean_ap = 0
    video_iou = []
    iou = 0
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
        iou += mean_iou
        
        for bounding_box in frames[index]:
                draw_rectangle_on_frame(frame, bounding_box)
        
        if generate_plot_video == True and index < end_plot:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(root_folder + "outputs/file%02d.png" % index)
            plt.clf()  
          
        if show_video == True:
             
            cv2.imshow('frame', frame)
            
            # wait for a key press     
            key = cv2.waitKey(1)
            

            # check if the user pressed the 'q' key to quit
            if key == ord('q'):
                break
    
        index += 1

    
    mean_ap = mean_ap /index
    mean_iou_final = iou / index
    print('Mean iou:', mean_iou_final)
    
    # release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    end_plot = min(end_plot, index)
    
    if generate_plot_video == True:
        os.chdir(root_folder + 'outputs/')
        subprocess.call([
            'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'traffic_video.mp4'
        ])
        for file_name in glob.glob(root_folder + 'outputs/*.png'):
            os.remove(file_name)
        
        for i in range(end_plot):
            plt.plot(list(range(start_plot, i)), video_iou[start_plot:i])
            plt.title('Noisy')
            plt.ylim(0,1)
            plt.xlim(0,end_plot)
            plt.ylabel('Mean IoU')
            plt.xlabel('Frames')
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(root_folder + "outputs/file%02d.png" % i)
            plt.clf()
        os.chdir(root_folder + 'outputs/')
        subprocess.call([
            'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'plot_video.mp4'
        ])
        for file_name in glob.glob(root_folder + 'outputs/*.png'):
            os.remove(file_name)
        
        videoClip = VideoFileClip(root_folder + "outputs/traffic_video.mp4")
        videoClip.write_gif(root_folder + "outputs/traffic_video.gif", fps=5)
        
        videoClip = VideoFileClip(root_folder + "outputs/plot_video.mp4")
        videoClip.write_gif(root_folder + "outputs/plot_video.gif", fps=5)
    else:
        start_plot = 0
        plt.plot(list(range(start_plot, end_plot)), video_iou[start_plot:end_plot])
        plt.title('SSD512')
        plt.ylim(0,1)
        plt.xlim(0,end_plot)
        plt.ylabel('Mean IoU')
        plt.xlabel('Frames')
        plt.show()
    
    