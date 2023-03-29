from src.sort import Sort
from dataset import *
import cv2
import pickle
import time
import sys
import io
import csv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task 2.2')
    parser.add_argument('--input_file_path', type=str, default='faster.pkl', help='Input file containing predictions (txt or pkl)')
    parser.add_argument('--display', type=bool, default=False, help='Save gif')
    parser.add_argument('--output_file_path', type=str, default="out.txt", help='Output prediction file path')
    parser.add_argument('--output_gif_file_path', type=str, default="out.gif", help='Output gif file path')
    parser.add_argument('--first_frame', type=int, default=0, help='First frame to analyse')
    parser.add_argument('--last_frame', type=int, default=2141, help='Last frame to analyse')
    args = parser.parse_args()
    #save args
    frame_start = args.first_frame
    frame_end = args.last_frame
    input_file = args.input_file_path 
    display = args.display
    output_file = args.output_file_path
    output_gif = args.output_gif_file_path

    #auxiliary function for saving gifs
    def fig2img(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    #create instance of SORT
    mot_kalman_tracker = Sort()


    def read_pkl_bboxes(input_pkl_path):
        """Load bounding box predictions from a pkl file containing the predicted bounding boxes.
        input_pkl_path: Input path of the pkl file containing gt and predicted bboxes. They are formatted in a two position array [bboxes_pred, bboxes_gt].
        This file can be generated saving bbox_pred and bbox_gt from task 1.1
        Outputs:
            pred_bboxes_proper_format: Predicted bboxes in the SORT format (List of lists where each box is defined as [xtl, ytl, xbr, ybr])
        
        """
        #load detections
        with open(input_pkl_path, 'rb') as handle:
            detections_dict = pickle.load(handle)


        bboxes_pred = detections_dict[0]
        bboxes_gt = detections_dict[1]
        detections = []
        index = frame_start
        pred_bboxes_proper_format = []

        for detection_index in bboxes_pred:
            bboxes_pred_list = bboxes_pred[detection_index]
            bboxes_gt_list = bboxes_gt[detection_index]
            curr_bboxes = []
            for bbox in bboxes_pred_list:
                bbox_proper_format = [ bbox['xtl'],  bbox['ytl'],bbox['xbr'],  bbox['ybr']]
                curr_bboxes.append(bbox_proper_format)
            pred_bboxes_proper_format.append(curr_bboxes)
            index +=1

        return pred_bboxes_proper_format

    def read_csv_bboxes(input_txt_path):
        """Load bounding box predictions from a txt file containing the predicted bounding boxes.
        input_txt_path: Input path of the pkl file containing the predicted bboxes. Each row is written in the MOT challenge format
        This file can be generated with the outputs of 2.1 (tracking ids are ignored)
        Outputs:
            pred_bboxes_proper_format: Predicted bboxes in the SORT format (List of lists where each box is defined as [xtl, ytl, xbr, ybr])
        
        """
        #load detections
        pred_bboxes_proper_format = []
        with open("output_retinanet.txt", 'r') as file:
            csvreader = csv.reader(file)
            prev_frame = 1
            curr_bboxes = []
            for row in csvreader:
                curr_frame = row[0]
                if curr_frame!=prev_frame:
                    pred_bboxes_proper_format.append(curr_bboxes)
                    curr_bboxes = []
                    prev_frame = curr_frame
                bbox_proper_format_temp = [ float(row[2]),  float(row[3]),float(row[4])+float(row[2]),  float(row[5])+float(row[3])]
                curr_bboxes.append(bbox_proper_format_temp)
        return pred_bboxes_proper_format

    if input_file.split('.')[-1] == "pkl":
        pred_bboxes_proper_format = read_pkl_bboxes(input_file)
    else:
        pred_bboxes_proper_format = read_csv_bboxes(input_file)

    #load sequence
    file_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
    out = []
    total_time = 0
    dataset_dicts = get_dicts('all', pretrained=True)

    bboxes_gt = {}
    bboxes_pred = {}
    video_frames_out =[]
    index = 0
    model_name = output_gif.split('.')[0]

    csv_path_kalman_dets = output_file
    # open the file in the write mode
    f_kalman_dets = open(csv_path_kalman_dets, 'w')

    writer_kalman_dets = csv.writer(f_kalman_dets)

    colours = np.random.rand(32,3) #used only for display
    for d in dataset_dicts:
        image_id = d["image_id"]
        if image_id<frame_start or image_id>frame_end:
            index = index+1
            #print("Skipping frame ", image_id)
            continue
        curr_detection = np.asarray(pred_bboxes_proper_format[index])
        curr_frame = cv2.imread(d["file_name"])
        
        start_time = time.time()
        trackers = mot_kalman_tracker.update(curr_detection)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        index = index+1
        
        out.append(trackers)
        if display:
            #save gif
            fig, ax = plt.subplots(1, 1)

            ax.imshow(curr_frame)
            ax.axis('off')
            ax.set_title(str(model_name) + 'detections')
        for j in range(np.shape(curr_detection)[0]):
            color = colours[j]
            coords = (curr_detection[j,0],curr_detection[j,1]), curr_detection[j,2], curr_detection[j,3]
            if display:
                ax.add_patch(patches.Rectangle((coords[0][0],coords[0][1]),coords[1]-coords[0][0],coords[2]-coords[0][1],fill=False,lw=3, ec = (1,0,0)))

        for d in trackers:
            d = d.astype(np.uint32)
            row = [image_id+1, d[4], d[0], d[1], d[2]-d[0], d[3]-d[1], 1,-1,-1,-1]
            writer_kalman_dets.writerow(row)
            curr_id = d[4]
            if display:
                ax.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=(0,1,0)))

        if display:
            frame_out = fig2img(fig)
            video_frames_out.append(frame_out)
            plt.close(fig)

    # close the csv files
    f_kalman_dets.close()
    if display:
        video_frames_out[0].save(output_gif, save_all=True, append_images=video_frames_out[1:], duration=30, loop=0)
        