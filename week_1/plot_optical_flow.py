import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
dataset_flow_path = '../datasets/data_stereo_flow/training/flow_noc/'
filename = '000045_10.png'
filename_path = dataset_flow_path+filename
dataset_image_path =  '../datasets/data_stereo_flow/training/image_0/'
image = '000045_10.png'
image_path = dataset_image_path+image
filename_path = dataset_flow_path+filename

np.set_printoptions(threshold=sys.maxsize)
def read_kitti_flow(filename):
    # In R, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # In G, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    # B = 0 for invalid flow (e.g., sky pixels)
    
    img = cv2.imread(filename,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH )

    flow = img.astype(np.float32)
    flow = flow[:,:,1:]
    flow = flow-2**15
    flow = flow/64
    #valid data is in the last channel (first one because opencv-> bgr instead of)
    valid_data = img[:,:,0] 
    
    return flow,valid_data  #WIP!! if you use this function, remmeber that its not in the proper order (first channel number, then v then u)

img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED )
out_flow, valid= read_kitti_flow(filename_path)

#store u v arrow directions
u = (out_flow[:, :, 1])
v = (out_flow[:, :, 0])

#subsample using step_size
step_size = 8
u = u[::step_size, ::step_size]
v = v[::step_size, ::step_size]
print("SHAPE",out_flow.shape)
h, w, coord = out_flow.shape

#already subsampled grid
nx = u.shape[1]
ny = u.shape[0]
X, Y = np.meshgrid(np.linspace(1, out_flow.shape[1],nx),np.linspace(1, out_flow.shape[0],ny))

print("U V X Y ", u.shape,v.shape,X.shape,Y.shape)

fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(img)
ax.quiver(X,Y, u, -v )  #kitti's coordinate origin is flipped for the y coordinate

# Hide grid lines
ax.grid(False)
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('optflow2.png', bbox_inches='tight', pad_inches=0, dpi = 100)
print("IN")
sys.exit()