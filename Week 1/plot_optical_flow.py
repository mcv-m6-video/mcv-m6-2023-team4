import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
from ipynb.fs.full.optical_flow_metrics import read_kitti_flow
dataset_flow_path = '../datasets/data_stereo_flow/training/flow_noc/'

filename = '000157_10.png'

dataset_flow_path = '../datasets/results/'
filename = 'LKflow_000045_10.png'
filename_path = dataset_flow_path+filename
dataset_image_path =  '../datasets/data_stereo_flow/training/image_0/'
image = '000045_10.png'
plot_save_path = 'optflow.png'
image_path = dataset_image_path+image
filename_path = dataset_flow_path+filename

np.set_printoptions(threshold=sys.maxsize)


def vector_to_rgb(angle, absolute):
    #from https://stackoverflow.com/questions/19576495/color-matplotlib-quiver-field-according-to-magnitude-and-direction
    global max_abs
    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                         absolute / max_abs, 
                                         absolute / max_abs))


img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED )
u,v, valid_data= read_kitti_flow(filename_path)
u = u*valid_data
v = v*valid_data

def plot_optical_flow_dense(u,v,plot_save_path):
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(u, v)
      
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
      
    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)
      
    ax.grid(False)
    # Hide axes ticks
    #ax.set_xlim(right = u.shape[1])
    #ax.set_ylim(bottom = u.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0, dpi = 200)
plot_optical_flow_color(u,v,img,plot_save_path) 
  
def plot_optical_flow_color(u,v,img,plot_save_path):
    #subsample using step_size
    step_size =10
    u_ssampled = u[::step_size, ::step_size]
    v_ssampled = v[::step_size, ::step_size]
    h, w = u_ssampled.shape

    #already subsampled grid
    nx = u_ssampled.shape[1]
    ny = u_ssampled.shape[0]
    X, Y = np.meshgrid(np.linspace(1, u.shape[1],nx),np.linspace(1, u.shape[0],ny))

    maxOF = max(np.max(u), np.max(v))
    angles = np.arctan2(v_ssampled, u_ssampled)
    lengths = np.sqrt(np.square(u_ssampled) + np.square(v_ssampled))
    max_abs = np.max(lengths)

    # color is the magnitude
    c = np.array(list(map(vector_to_rgb, 2 * np.pi * lengths.flatten() / max_abs*0.7, 
                                        max_abs * np.ones_like(lengths.flatten()))))

    fig, ax = plt.subplots(figsize=(40,40))
    ax.axis("off")
    ax.imshow(img, cmap='gray')
    q = ax.quiver(X,Y, u_ssampled, -v_ssampled,scale=maxOF*8,color = c,width=0.001)  #kitti's coordinate origin is flipped for the y coordinate

    # Hide grid lines
    ax.grid(False)
    # Hide axes ticks
    #ax.set_xlim(right = u.shape[1])
    #ax.set_ylim(bottom = u.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0, dpi = 200)

