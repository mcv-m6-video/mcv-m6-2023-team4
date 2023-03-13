from src.parser import *
from src.noisy import *

def main():
    frames = parse_xml('../datasets/ai_challenge_s03_c010-full_annotation.xml')
    frames_noisy, frames_all = generate_noisy_annotations(frames, th_dropout=0.3, th_generate=0.3, mean=0, std=10)
    parse_video('../datasets/AICity_data/train/S03/c010/vdo.avi', frames_all)


if __name__ == "__main__":
    main()

