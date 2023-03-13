from src.parser import *
from src.noisy import *

def main():
    root_folder = 'c:/Users/Marcos/Desktop/Master Computer Vision/M6/Project/Work/mcv-m6-2023-team4/datasets/'
    # root_folder = '../datasets/'
    frames = parse_xml(root_folder + '/ai_challenge_s03_c010-full_annotation.xml')
    
    
    frames_noisy = parse_detection_txt('C:/Users/Marcos/Desktop/Master Computer Vision/M6/Project/Work/mcv-m6-2023-team4/datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')
    frames_all = copy.deepcopy(frames)
    for index, frame in frames_noisy.items():
        for bbox in frame:
            frames_all[index].append(bbox)
                
    
    # frames_noisy, frames_all = generate_noisy_annotations(frames, th_dropout=0.3, th_generate=0.3, mean=0, std=10)
    parse_video(root_folder + '/AICity_data/train/S03/c010/vdo.avi', frames_all, False, True)
    


if __name__ == "__main__":
    main()

