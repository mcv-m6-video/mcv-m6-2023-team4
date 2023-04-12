import sys
import cv2
import matplotlib.pyplot as plt
import os
import subprocess
import glob
from moviepy.editor import VideoFileClip
import numpy as np
import csv
from shapely.geometry import Polygon
import math

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
    
    
    # Parse groundtruth from video
    def parse_detection_txt(self, file_path, roi_path=None, remove_parked_cars=True):
        if roi_path is not None:
            roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            
            _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
            
            # first_frame = self.get_first_frame()
            # plt.imshow(roi, cmap='gray')
            # plt.show()
            # plt.imshow(first_frame[:,:,::-1])
            # plt.show()
            # first_frame_parsed = first_frame.copy()
            # first_frame_parsed[:, :, 0] = first_frame_parsed[:, :, 0] * (roi/255)
            # first_frame_parsed[:, :, 1] = first_frame_parsed[:, :, 1] * (roi/255)
            # first_frame_parsed[:, :, 2] = first_frame_parsed[:, :, 2] * (roi/255)
            # first_frame_parsed = first_frame_parsed.astype(np.uint8)
            # plt.imshow(first_frame_parsed[:,:,::-1])
            # plt.show()
            # sys.exit()
        
        # Format [frame, -1, left, top, width, height, conf, -1, -1, -1].
        detections = {}
        
        box_frame = -1
        
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                box_frame, id, left, top, width, height, conf, _, _, _ = map(float, row)
                box_frame = int(box_frame)
                box_frame -= 1
                left, top, width, height, conf = map(float, [left, top, width, height, conf])
                
                if box_frame not in detections:
                    detections[box_frame] = []
                    
                bbox = {'id': int(id),
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
                    'confidence': conf}
                
                
                    
                if roi_path is not None and not self.inside_roi(roi, bbox):
                    # print(self.inside_roi(roi, bbox))
                    continue
                    
                detections[box_frame].append(bbox)
        
        if remove_parked_cars and box_frame != -1:
            detections = self.remove_parked_cars(detections, box_frame)
            
        return detections
    
    # Return if bounding box is inside ROI or not
    def inside_roi(self, roi, bbox):
        height, width = roi.shape
        c_x, c_y = self.bbox_centroid([bbox['xtl'], bbox['ytl'], bbox['xbr'], bbox['ybr']])
        if c_y > height or c_x > width:
            return True
        
        c_x = int(np.floor(c_x))
        c_y = int(np.floor(c_y))
        
        return roi[c_y, c_x] == 255
    
    # Remove parked cars
    # For method distance a good threshold is 4
    # For method iou a good threshold is 0.9
    def remove_parked_cars(self, detections, num_frames, method='distance', th=4):
        detections_numpy = self.detections_dict_to_numpy(detections)
        
        # Num object ids, bbox data and num frames
        num_object_ids = int(np.max(detections_numpy[:, 1]))
        time_cube = np.ones((num_object_ids, num_frames, 6)) * -1
        
        # Create a time cube for the video
        for frame in detections:
            for bbox in detections[frame]:
                time_cube[bbox['id']-1, int(frame)-1, :] = [int(frame), bbox['id'], bbox['xtl'], bbox['ytl'], bbox['xbr'],
                                                            bbox['ybr']]

        object_ids_to_remove = []
        
        # Diferent ways to remove an object from frame
        for object_id in range(num_object_ids):
            accumulator = []
            
            # Get only objects in contiguous times
            object_time_cube = time_cube[object_id, :, :]
            
            # Remove frames where object does not appear
            object_time_cube = object_time_cube[~(object_time_cube == -1).any(1),:]

            num_boxes = object_time_cube.shape[0]
            
            if num_boxes == 0:
                continue
            
            
            # If iou too similar for all the time an object appears remove it   
            for index, bbox1 in enumerate(object_time_cube):
                if index+1 == num_boxes:
                    break
                
                bbox2 = object_time_cube[index+1, :]
                
                if method == 'distance':
                    # Get euclidean distance between one bounding box centroid and the next bounding box centroid
                    # We accumulate distance in order to check the speed of the object, if object speed is too large it is moving             
                    accumulator.append(self.bbox_distance(bbox1[2:], bbox2[2:]))
                elif method == 'iou':                
                    # Get iou between one bounding box and the next
                    accumulator.append(self.bbox_iou(bbox1[2:], bbox2[2:]))
            
            accumulator = np.array(accumulator)
            
            # If the distance between the initial position of the object and the final is too big, obviously it is not static
            if method == 'distance':
                bbox1 = object_time_cube[0, :]
                bbox2 = object_time_cube[-1, :]
                
                if self.bbox_distance(bbox1[2:], bbox2[2:]) > 300:
                    continue
            
            if np.sum(accumulator) / len(accumulator) < th:
                object_ids_to_remove.append(object_id+1)
            
            # print('Object id:', object_id+1)
            # print(np.sum(accumulator) / len(accumulator))
        
        # Now remove the parked objects based on the previous method
        detections_without_parked = {}
        
        for frame in detections:
            for bbox in detections[frame]:
                if bbox['id'] in object_ids_to_remove:
                    # print(str(bbox['id'])+' was removed')
                    continue
                
                if frame not in detections_without_parked:
                    detections_without_parked[frame] = []
                
                detections_without_parked[frame].append(bbox)
        
        return detections_without_parked
    
    
    def bbox_iou(self, bboxA, bboxB):
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
    
    # Get bounding box centroid
    def bbox_centroid(self, bbox):
        c_x = (bbox[0] + bbox[2]) / 2
        c_y = (bbox[1] + bbox[3]) / 2
        return (c_x, c_y)
    
    # Get distance between two bounding box centroids
    def bbox_distance(self, bboxA, bboxB):
        c1_x, c1_y = self.bbox_centroid(bboxA)
        c2_x, c2_y = self.bbox_centroid(bboxB)

        # Calculate the Euclidean distance between the centroids
        distance = math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)     
        return distance
    
    def detections_dict_to_numpy(self, detections):
        detections_numpy = []
        for frame in detections:
            for bbox in detections[frame]:
                x, y, z = -1, -1, -1  # No information about x, y, z
                
                detections_numpy.append([int(frame), bbox['id'], bbox['xtl'], bbox['xtl'], bbox['ytl'],
                                         bbox['xbr'] - bbox['xtl'], bbox['ybr'] - bbox['ytl'], 1, x, y, z])
                
        detections_numpy = np.array(detections_numpy, dtype=np.float64)
        return detections_numpy        
    
    # Create bounding box on a frame
    def draw_rectangle_on_frame(self, frame, bounding_box, bbox_color=(255, 0, 0)):
        xtl, ytl, xbr, ybr = map(int, [bounding_box['xtl'], bounding_box['ytl'], bounding_box['xbr'], bounding_box['ybr']])
        # draw rectangle on image
        if bounding_box['predicted']:
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), bbox_color, 2)    
            # self.put_text(frame, str(round(bounding_box['iou'], 2)), (xtl, ytl), text_color=(0, 0, 0))
        elif bounding_box['parked']:
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), bbox_color, 2)
        else:
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), bbox_color, 2)
        
        if bounding_box['id'] != -1:
            self.put_text(frame, str(bounding_box['id']), (xtl, ytl), bbox_width=xbr-xtl, bbox_color=bbox_color, text_color=(56, 56, 56))
            
    # Put text on a part of a video
    def put_text(self, frame, text, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN, font_scale=2, font_thickness=3, bbox_color=(13,255,255),
                 bbox_width = 12,
                 text_color=(0, 255, 0)):
        x, y = pos
        try:
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            # sub_img = frame[y:y+text_h, x:x+text_w]
            sub_img = frame[y:y+text_h, x:x+bbox_width]
            background_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            background_rect[:,:,0] = bbox_color[0]
            background_rect[:,:,1] = bbox_color[1]
            background_rect[:,:,2] = bbox_color[2]
            res = cv2.addWeighted(sub_img, 0.5, background_rect, 0.5, 1.0)
            
            frame[y:y+text_h, x:x+bbox_width] = res
            # frame[y:y+text_h, x:x+text_w] = res
            
            cv2.putText(frame, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
        except:
            pass
    
    
    def get_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        
        index = 0
        while True:
            ret, frame = cap.read()
            return frame
            index += 1
    
    # Read video from a file
    # show_video: If True the read video will be shown on a new display
    # create_video_annot: Parses video to mp4 with annotations
    def parse_video(self, min_frame=0, end_frame=np.Inf, show_video=False, create_video_annot=False, groundtruth={}, annotations={}, create_gif=False):
        cap = cv2.VideoCapture(self.video_path)
        
        
        index = 0
        while True:
            ret, frame = cap.read()
            
            if not ret or index > end_frame:
                break
            
            if index in groundtruth:
                for bounding_box in groundtruth[index]:
                    self.draw_rectangle_on_frame(frame, bounding_box, bbox_color=(38, 38, 38))
                            
            if index in annotations:
                for bounding_box in annotations[index]:
                    self.draw_rectangle_on_frame(frame, bounding_box, bbox_color=(13, 252, 252))
                
            if show_video:
                cv2.imshow('frame', frame)
                
                # wait for a key press     
                key = cv2.waitKey(1)
                
                # check if the user pressed the 'q' key to quit
                if key == ord('q'):
                    break
            
            
            if (create_gif or create_video_annot) and index >= min_frame:
                cv2.imwrite('./outputs/video/file%02d.png' % (index-min_frame), frame)
                
            index += 1
            
        if create_gif or create_video_annot:
            # video_dirname = os.path.dirname(self.video_path)
            # video_filename = os.path.basename(self.video_path)
            parsed_filename = os.path.dirname(self.video_path).replace('.','').replace('/', '_')
            
            if os.path.isfile('./outputs/video/%s.mp4'%parsed_filename):
                os.remove('./outputs/video/%s.mp4'%parsed_filename)
            
            os.chdir('./outputs/video/')
            subprocess.call([
                'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                '%s.mp4'%parsed_filename
            ])
            os.chdir('../../')
            for file_name in glob.glob('./outputs/video/*.png'):
                os.remove(file_name)
                
            if create_video_annot == True:
                videoClip = VideoFileClip('./outputs/video/%s.mp4'%parsed_filename)
                if create_gif == True:
                    videoClip.write_gif('./outputs/video/%s.gif'%parsed_filename, fps=3)
                videoClip.close()
        
        return
        sys.exit()
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