import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from opt_flow import *
import pandas as pd
import os

imagename1 = '000045_10.png'
imagename2 = '000045_11.png'

dataset_flow_path = '../datasets/results/LKflow_000045_10.png'
dataset_image_path =  '../datasets/data_stereo_flow/training/image_0/'

filename_path = dataset_flow_path
image_path = dataset_image_path + imagename1

img1 = cv2.imread(dataset_image_path + imagename1, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(dataset_image_path + imagename2, cv2.IMREAD_UNCHANGED)

def optical_flow_metrics(opticalflow_pred, filename_gt):

    dataset_flow_path_gt = '../datasets/data_stereo_flow/training/flow_noc/'
    filename_path_gt = dataset_flow_path_gt + filename_gt

    u_gt, v_gt, valid_data_gt = read_kitti_flow(filename_path_gt)
    u_gt = u_gt * valid_data_gt
    v_gt = v_gt * valid_data_gt

    u_pred = opticalflow_pred[:, :, 0]
    v_pred = opticalflow_pred[:, :, 1]

    msen, msen_distances = MSEN(u_pred, v_pred, u_gt, v_gt, valid_data_gt)
    pepn, pepn_distances = PEPN(u_pred, v_pred, u_gt, v_gt, valid_data_gt)
    
    print('MSEN for image', filename_gt + ":", round(msen,4))
    print('PEPN for image', filename_gt + ":", str(round(pepn,4)) + "%")
    
    return msen, pepn

def optical_flow(block_side_length=50, area_of_search_size=40, step_size=5, error_function="NCC", idx=0):

    motion_vectors = []
    block_coordinates_frame1 = []

    for i in range(0, img1.shape[0], block_side_length):
        for j in range(0, img1.shape[1], block_side_length):

            max_y = i + block_side_length
            max_x = j + block_side_length

            max_y = max_y if max_y < img1.shape[0] else img1.shape[0] - 1
            max_x = max_x if max_x < img1.shape[1] else img1.shape[1] - 1

            block_frame1 = img1[i:max_y, j:max_x]
            block_coordinates_frame1.append([j, i])

            area_of_search_coords = [max(0, i-area_of_search_size), max(0, j-area_of_search_size), min(img1.shape[0], max_y+area_of_search_size), min(img1.shape[1], max_x+area_of_search_size)]

            best_metric_val = None
            best_block = None

            for m in range(area_of_search_coords[0], area_of_search_coords[2], step_size):
                for n in range(area_of_search_coords[1], area_of_search_coords[3], step_size):

                    max_y_frame2 = m + block_side_length
                    max_x_frame2 = n + block_side_length

                    max_y_frame2 = max_y_frame2 if max_y_frame2 < img2.shape[0] else img2.shape[0] - 1
                    max_x_frame2 = max_x_frame2 if max_x_frame2 < img2.shape[1] else img2.shape[1] - 1
                    
                    block_frame2 = img2[m:max_y_frame2, n:max_x_frame2]

                    if block_frame1.shape[0] == block_frame2.shape[0] and block_frame1.shape[1] == block_frame2.shape[1]:

                        ssd = np.sum(np.square(block_frame1 - block_frame2))
                        sad = np.sum(np.abs(block_frame1 - block_frame2))
                        ncc = np.sum((block_frame1 - np.mean(block_frame1)) * (block_frame2 - np.mean(block_frame2))) / (np.std(block_frame1) * np.std(block_frame2) * block_frame1.size + 1e-7)

                        if error_function == "NCC":
                            if best_metric_val is None or ncc > best_metric_val:
                                best_metric_val = ncc
                                best_block = [m, max_y_frame2, n, max_x_frame2]
                        elif error_function == "SSD":
                            if best_metric_val is None or ssd < best_metric_val:
                                best_metric_val = ssd
                                best_block = [m, max_y_frame2, n, max_x_frame2]
                        elif error_function == "SAD":
                            if best_metric_val is None or sad < best_metric_val:
                                best_metric_val = sad
                                best_block = [m, max_y_frame2, n, max_x_frame2]


            if best_block is not None:
              motion_vector = np.array(best_block) - np.array([i, max_y, j, max_x])
            else:
              motion_vector = np.array([0, 0, 0, 0])
              
            motion_vectors.append([motion_vector[2], motion_vector[0]])


    opticalflow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
    block_coordinates_frame1 = np.array(block_coordinates_frame1)

    for i in range(len(motion_vectors)):
        block_x, block_y = block_coordinates_frame1[i]
        u, v = motion_vectors[i]
        opticalflow[block_y:block_y+block_side_length, block_x:block_x+block_side_length, 0] = u
        opticalflow[block_y:block_y+block_side_length, block_x:block_x+block_side_length, 1] = v

    # Bilinear interpolation for the missing pixels
    x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
    interpolated_opticalflow = np.zeros_like(opticalflow)
    interpolated_opticalflow[..., 0] = np.interp(x.flatten(), block_coordinates_frame1[:, 0], np.array(motion_vectors)[..., 0].flatten()).reshape([img1.shape[0], img1.shape[1]])
    interpolated_opticalflow[..., 1] = np.interp(y.flatten(), block_coordinates_frame1[:, 1], np.array(motion_vectors)[..., 1].flatten()).reshape([img1.shape[0], img1.shape[1]])


    plot_save_path = 'Results/optflow_week4' + str(idx) + '.png'

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    u = opticalflow[:, :, 0]
    v = opticalflow[:, :, 1]

    #plot_optical_flow_color(u,v,img,plot_save_path)

    plot_save_path = 'Results/optflow_week4_interpolated' + str(idx) + '.png'

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    u = interpolated_opticalflow[:, :, 0]
    v = interpolated_opticalflow[:, :, 1]

    #plot_optical_flow_color(u,v,img,plot_save_path)

    #msen, pepn = optical_flow_metrics(opticalflow, '000045_10.png')
    msen, pepn = optical_flow_metrics(interpolated_opticalflow, '000045_10.png')

    return msen, pepn


if "task_1.1.csv" not in os.listdir("Results/"):
    df = pd.DataFrame(columns=["idx", "Area of search", "Block size", "Step size", "Error function", "MSEN", "PEPN"])
    df.to_csv("Results/task_1.1.csv")

idx = 0
for area_of_search_size in np.arange(10, 300, 50):
    for block_size in np.arange(5, 220, 25):
        for step_size in np.arange(1, 10, 1):
            for error_function in ["NCC", "SAD", "SSD"]:
                msen, pepn = optical_flow(block_side_length=block_size, area_of_search_size=area_of_search_size, step_size=step_size, error_function=error_function, idx=idx)

                idx = idx + 1

                df = pd.read_csv("Results/task_1.1.csv")
                df = pd.concat([df, pd.DataFrame([{"idx": idx, "Area of search": area_of_search_size, "Block size": block_size, "Step size": step_size, "Error function": error_function, "MSEN": msen, "PEPN": pepn}])], axis=0, ignore_index=True)
                df.to_csv("Results/task_1.1.csv", index=False)