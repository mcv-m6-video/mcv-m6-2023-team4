from src.parser import *
from src.noisy import *
from src.utils import *

def main():
    root_folder = '../datasets/'
    # root_folder = '../datasets/'
    frames = parse_xml(root_folder + 'ai_challenge_s03_c010-full_annotation.xml')
    
    #frames_noisy = parse_detection_txt('../datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    #frames_all = merge_frames(frames,frames_noisy)
                
    
    frames_noisy, frames_all = generate_noisy_annotations(frames, th_dropout=0.3, th_generate=0.3, mean=0, std=10)
    parse_video(root_folder + '/AICity_data/train/S03/c010/vdo.avi', frames_all, show_video=True, without_confidence=True)
    


if __name__ == "__main__":
    main()

