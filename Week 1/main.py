from src.parser import *
from src.noisy import *
from src.utils import *
from moviepy.editor import VideoFileClip


def main():
    # root_folder = 'c:/Users/Marcos/Desktop/Master Computer Vision/M6/Project/Work/mcv-m6-2023-team4/'
    root_folder = '../'

    frames = parse_xml(root_folder + '/datasets/ai_challenge_s03_c010-full_annotation.xml')
    
    frames_noisy = parse_detection_txt(root_folder + '/datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    frames_all = merge_frames(frames,frames_noisy)
    
    
    frames_noisy, frames_all = generate_noisy_annotations(frames, th_dropout=0, th_generate=0, mean=0, std=0)
    parse_video(root_folder + '/datasets/AICity_data/train/S03/c010/vdo.avi', frames_all, show_video=True, without_confidence=True, generate_plot_video=False)
    
    rec, prec, ap, iou, mean_iou = voc_eval(frames, frames_noisy)

    print(mean_iou)
    print(ap)


if __name__ == "__main__":
    main()

