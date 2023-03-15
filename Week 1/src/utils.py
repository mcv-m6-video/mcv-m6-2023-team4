import cv2
from IPython.display import Video, display
import numpy as np
import copy

def put_text(frame, text, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN, font_scale=2, font_thickness=3, text_color=(0, 255, 0)):
    x, y = pos
    try:
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        sub_img = frame[y:y+text_h, x:x+text_w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        
        frame[y:y+text_h, x:x+text_w] = res
        
        cv2.putText(frame, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    except:
        pass

def draw_rectangle_on_frame(frame, bounding_box):
    xtl, ytl, xbr, ybr = map(int, [bounding_box['xtl'], bounding_box['ytl'], bounding_box['xbr'], bounding_box['ybr']])
    # draw rectangle on image
    if bounding_box['predicted']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)    
        put_text(frame, str(round(bounding_box['iou'], 2)), (xtl, ytl), text_color=(0, 0, 0))
    elif bounding_box['parked']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

def merge_frames(frames, frames_noisy):
    frames_all = copy.deepcopy(frames)
    for index, frame in frames_noisy.items():
        for bbox in frame:
            frames_all[index].append(bbox)

    return frames_all

