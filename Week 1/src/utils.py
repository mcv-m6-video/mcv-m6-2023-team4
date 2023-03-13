import cv2
from IPython.display import Video, display


def draw_rectangle_on_frame(frame, bounding_box):
    xtl, ytl, xbr, ybr = map(int, [bounding_box['xtl'], bounding_box['ytl'], bounding_box['xbr'], bounding_box['ybr']])
    print(bounding_box)
    # draw rectangle on image
    if bounding_box['noisy']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
    elif bounding_box['parked']:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)



