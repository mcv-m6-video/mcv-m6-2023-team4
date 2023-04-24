from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
from opt_flow import plot_optical_flow_color, optical_flow_metrics
import sys, os
# from __future__ import unicode_literals
import numpy as np
import cv2
import time
import argparse
import pyflow   #install it by doing: pip install ./pyflow


def compute_OF_pyflow(im1, im2):
    
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    

    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    valid_data =  np.ones((im1.shape[0], im1.shape[1]))
    flow = np.concatenate((u[..., None], v[..., None], valid_data[..., None]), axis=2)

    sys.stdout = sys.__stdout__
        
    return flow