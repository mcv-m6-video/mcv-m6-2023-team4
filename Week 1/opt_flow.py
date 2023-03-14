import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
import seaborn as sns

def read_kitti_flow(filename):
    #reads the optical flow from the KITTI dataset
    # In R, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # In G, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # B = 0 for invalid flow (e.g., sky pixels)
    
    img = cv2.imread(filename,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH )

    flow = img.astype(np.float32)
    flow = flow[:,:,1:]
    #undo quantization
    flow = flow-2**15
    flow = flow/64
    #valid data is in the last channel (first one because opencv-> bgr instead of)
    valid_data = img[:,:,0]
    
    #store u v arrow directions
    u = (flow[:, :, 1])
    v = (flow[:, :, 0])
    return u,v,valid_data  

def MSEN(u_pred, v_pred, u_gt, v_gt, valid_data_gt):
    distances = np.sqrt(np.power(u_pred - u_gt, 2) + np.power(v_pred - v_gt, 2))
    return np.mean(distances[valid_data_gt!=0]),distances

def PEPN(u_pred, v_pred, u_gt, v_gt, valid_data_gt, th=3):
    distances = np.sqrt(np.power(u_pred - u_gt, 2) + np.power(v_pred - v_gt, 2))
    return (np.sum(distances[valid_data_gt!=0] > th) / np.sum(valid_data_gt!=0)) * 100, distances

def optical_flow_metrics(filename_pred, filename_gt):
    
    dataset_flow_path_gt = '../datasets/data_stereo_flow/training/flow_noc/'
    filename_path_gt = dataset_flow_path_gt + filename_gt

    dataset_flow_path_pred = '../datasets/results/'
    filename_path_pred = dataset_flow_path_pred + filename_pred

    u_gt, v_gt, valid_data_gt = read_kitti_flow(filename_path_gt)
    u_gt = u_gt * valid_data_gt
    v_gt = v_gt * valid_data_gt

    u_pred, v_pred, valid_data_pred = read_kitti_flow(filename_path_pred)
    u_pred = u_pred * valid_data_pred
    v_pred = v_pred * valid_data_pred

    msen, msen_distances = MSEN(u_pred, v_pred, u_gt, v_gt, valid_data_gt)
    pepn, pepn_distances = PEPN(u_pred, v_pred, u_gt, v_gt, valid_data_gt)
    
    print('MSEN for image', filename_gt + ":", round(msen,4))
    print('PEPN for image', filename_gt + ":", str(round(pepn,4)) + "%")
    return msen_distances, pepn_distances, valid_data_gt
def plot_MSEN(msen_array, image_name):
    im = plt.matshow(msen_array, aspect='auto') 
    plt.colorbar(im)
    plt.title("Pixel-wise MSEN (image "+str(image_name)+")")
    plt.show()
def MSEN_histogram(msen_array, valid_data_gt, image_name, ylim, binwidth=0.5):
    sns.histplot(data=msen_array[valid_data_gt!=0].flatten(),stat="probability", kde=True, binwidth=binwidth)
    plt.title("Normalized MSEN histogram (image "+str(image_name)+ " )")
    plt.ylim(0, ylim)

def plot_optical_flow_dense(u,v,img,plot_save_path):
    #The angle (direction) of flow by hue is visualized and the distance (magnitude) of flow by the value of HSV color representation.
    #The strength of HSV is always set to a maximum of 255 for optimal visibility.
    # Referenced from opencv docs
    mask = np.zeros_like((img))
    # Sets image saturation to maximum
    mask[..., 1] = 255
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(u, -v)
      
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
      
    
    fig, ax = plt.subplots(figsize=(40,40))
    
    plt.imshow(rgb)
    ax.grid(False)
    ax.set_xticks([])
    ax.axis("off")
    ax.imshow(rgb, cmap='gray')
    ax.set_yticks([])
    # Opens a new window and displays the output frame
    cv2.imwrite("dense"+plot_save_path, rgb)

global max_abs
def plot_optical_flow_color(u,v,img,plot_save_path):
    def vector_to_rgb(angle, absolute):
        #from https://stackoverflow.com/questions/19576495/color-matplotlib-quiver-field-according-to-magnitude-and-direction

        # normalize angle
        angle = angle % (2 * np.pi)
        if angle < 0:
            angle += 2 * np.pi

        return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                            absolute / max_abs, 
                                            absolute / max_abs))
    #subsample using step_size
    step_size =10
    u_ssampled = u[::step_size, ::step_size]
    v_ssampled = v[::step_size, ::step_size]
    h, w = u_ssampled.shape

    #already subsampled grid
    nx = u_ssampled.shape[1]
    ny = u_ssampled.shape[0]
    X, Y = np.meshgrid(np.linspace(1, u.shape[1],nx),np.linspace(1, u.shape[0],ny))

    max_uv = max(np.max(u), np.max(v))
    angles = np.arctan2(v_ssampled, u_ssampled)
    lengths = np.sqrt(np.square(u_ssampled) + np.square(v_ssampled))
    max_abs = np.max(lengths)

    # color is the magnitude
    c = np.array(list(map(vector_to_rgb, 2 * np.pi * lengths.flatten() / max_abs*0.7, 
                                        max_abs * np.ones_like(lengths.flatten()))))

    fig, ax = plt.subplots(figsize=(40,40))
    ax.axis("off")
    ax.imshow(img, cmap='gray')
    q = ax.quiver(X,Y, u_ssampled, -v_ssampled,scale=max_uv*8,color = c,width=0.001)  #kitti's coordinate origin is flipped for the y coordinate

    # Hide grid lines
    ax.grid(False)
    # Hide axes ticks
    #ax.set_xlim(right = u.shape[1])
    #ax.set_ylim(bottom = u.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

    #plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0, dpi = 200)