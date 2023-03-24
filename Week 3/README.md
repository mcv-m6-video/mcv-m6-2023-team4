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
