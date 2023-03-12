import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
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

sys.exit()
