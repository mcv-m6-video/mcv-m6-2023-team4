import numpy as np
import copy
from src.parser import *
from src.utils import *

# TODO: FIX GENERATED BOUNDING BOXES, THEY ARE GENERATED ON THE LEFT CORNER ONLY? WHY
def generate_random_bbox(img_width, img_height, min_size, max_width, max_height):
    
    dp = np.random.randint(50,250)
    xtl = max(10, 1900*np.random.random() - dp)
    ytl = max(10, 1000*np.random.random() - dp)
    xbr =  xtl + dp
    ybr =  ytl + dp
    
    
    #Move the bbox generated randomly
    #dx = np.random.randint(0, 750)
    #dy = np.random.randint(0, 750)

    generated_bbox = {
                        'track_id': None,
                        'xtl': xtl,
                        'ytl': ytl,
                        'xbr': xbr,
                        'ybr': ybr,
                        'outside': None,
                        'occluded': None,
                        'keyframe': None,
                        'parked': None,
                        'predicted': True,
                        'confidence': float(0)
                    }
    #print(generated_bbox)
    return generated_bbox

def move_random_bbox(bounding_box, mean, std):
    # Move box / make it bigger
    dx1, dy1, dx2, dy2 = mean + std * np.random.randn(4)
    bbox_moved = {
            'xtl': bounding_box['xtl'] + dx1,
            'ytl': bounding_box['ytl'] + dy1,
            'xbr': bounding_box['xbr'] + dx2,
            'ybr': bounding_box['ybr'] + dy2,
            'outside': None,
            'occluded': None,
            'keyframe': None,
            'parked': None,
            'predicted': True,
            'confidence': float(0)
        }
    return bbox_moved


def generate_noisy_annotations(frames, th_dropout, th_generate, mean, std):
    frames_all = copy.deepcopy(frames)
    frames_noisy = {}
    
    for index, frame in frames.items():                  
        frames_noisy[index] = []
        for bounding_box in frame:

            dropout = np.random.uniform(0,1)
            if dropout > th_dropout:
                bbox_moved = move_random_bbox(bounding_box, mean, std)
                frames_noisy[index].append(bbox_moved)
                frames_all[index].append(bbox_moved)
            
        # In each frame we generate or not a not annotated bounding box
        generate_probability = np.random.uniform(0,1)
        
        if generate_probability > th_generate:
            bounding_boxes_gen_num = np.random.randint(1, 4)
            for i in range(bounding_boxes_gen_num):
                random_bbox = generate_random_bbox(1920, 1080, 40, np.random.randint(250, 400),np.random.randint(250,400))
                
                frames_noisy[index].append(random_bbox)
                frames_all[index].append(random_bbox)
        
    return frames_noisy, frames_all