# How to run

## Task 1.1


## Task 1.2

Libraries have been installed from the code repositories following their respective tutorials. See the task_1_2.ipynb notebook for implementation links to the off-the-shelf methods and a visualisation/library setup demo with the image 000045_10.png.

## Task 1.3
See the task1_3.ipynb for implementation. The TrackEval folder containing the obtained results in MOT16 challenge format has been added. Run the following script to obtain the results
                python TrackEval/scripts/run_mot_challenge.py --BENCHMARK lucas-kanade-median --DO_PREPROC False 

## Task 2

See the task_2.ipynb notebook to generate the video.  The outputted predictions are in /outputs and data_simple (TrackEval format). See task_2_generate_hotas, task_2_inference_detectron and task_2_train_detectron python files to train and detect tracking objects. For finetuning on the different sequences use task_2_train_detectron python file.
