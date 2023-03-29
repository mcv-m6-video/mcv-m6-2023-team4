# How to run

## Task 1.1

1. Run the file that reads the video and saves each frame as an image:

```cmd
python dataset.py
```

2. For each network and threshold you want to use, run the command:

```cmd
python task_1.1.py --network faster_RCNN --th 0.5

python task_1.1.py --network faster_RCNN --th 0.7

python task_1.1.py --network mask_RCNN --th 0.5

python task_1.1.py --network mask_RCNN --th 0.7
```

Each result will be in the folder **Results/Task_1.1/*NETWORK_NAME*/*CONFIDENCE_SCORE*/**, for example for the first command the result will be in the folder: **Results/Task_1.1/faster_RCNN/0.5/**.

Alternatively, you can run the 4 jobs in this folder, using *sbatch*. They correspond to the 4 configurations indicated in the filename itself:

```cmd
sbatch --gres gpu:1 -n 10 job_task1.1_fasterrcnn_0.5

sbatch --gres gpu:1 -n 10 job_task1.1_fasterrcnn_0.7

sbatch --gres gpu:1 -n 10 job_task1.1_maskrcnn_0.5

sbatch --gres gpu:1 -n 10 job_task1.1_maskrcnn_0.7
```

## Task 1.2

See the annotations.ipynb notebook. The final annotations are on the file Anotation_AICITY_S05_C010.xml

## Task 1.3

Run the file that fine-tunes and then evaluates the network you specify using the parameter *--network*. You can also specify the initial learning rate using the parameter *--lr*. The default one is 1e-3:

```cmd
python task_1.3.py --network faster_RCNN --lr 1e-3
```

## Task 1.4

Run the file that fine-tunes and then evaluates the network you specify using the parameter *--network* using cross-validation. You can also specify the initial learning rate using the parameter *--lr*. The default one is 1e-3. For cross-validation you have to use the parameter *--cross-validation--*. There are several options such as None, Normal and Random

```cmd
python task_1.4.py --network faster_RCNN --lr 1e-3 --cross-validation=Normal
```

## Task 2.1
See the task_2_1.ipynb notebook to generate the video.  The outputted predictions are in /outputs and data_simple (TrackEval format). 

## Task 2.2

See the task_2_2.ipynb notebook for generating gif visualisations. The outputted predictions are in /outputs and data_kalman (TrackEval format). Additionally, a .py has been provided. Input file of the detections must be provided (MOT challenge format).See --help for options.
```cmd
python task_2.2.py --help
```

```cmd
python task_2.2.py --input_file input.txt 
```

## Task 2.3

SORT results can be evaluated by replacing the contents of folder data with the contents of the provided /outputs/data_kalman and running the following in TrackEval/scripts:
```cmd
python run_mot_challenge.py  --BENCHMARK Faster-RCNN --DO_PREPROC False
```

```cmd
python run_mot_challenge.py  --BENCHMARK MASK-RCNN --DO_PREPROC False
```

```cmd
python run_mot_challenge.py  --BENCHMARK RetinaNet --DO_PREPROC False
```

```cmd
python run_mot_challenge.py  --BENCHMARK Faster-RCNN-finetuned --DO_PREPROC False
```