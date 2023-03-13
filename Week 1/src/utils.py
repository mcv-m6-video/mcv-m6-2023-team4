import cv2
from IPython.display import Video, display


def draw_rectangle_on_frame(frame, bounding_box):
    xtl, ytl, xbr, ybr = map(int, [bounding_box['xtl'], bounding_box['ytl'], bounding_box['xbr'], bounding_box['ybr']])
    # draw rectangle on image
    if bounding_box['predicted']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)    
        cv2.putText(frame, str(round(bounding_box['iou'], 2)), (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)    
    elif bounding_box['parked']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)



