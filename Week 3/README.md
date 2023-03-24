# How to run

## Task 1.1

1. Run the file that reads the video and saves each frame as an image:

```cmd
python dataset.py
```

2. For each network, run a job using *sbatch*:

```cmd
sbatch --gres gpu:1 -n 10 job_task1.1_fasterrcnn_0.5

sbatch --gres gpu:1 -n 10 job_task1.1_fasterrcnn_0.7

sbatch --gres gpu:1 -n 10 job_task1.1_maskrcnn_0.5

sbatch --gres gpu:1 -n 10 job_task1.1_maskrcnn_0.7
```

Each result will be in the folder **Results/Task_1.1/*NETWORK_NAME*/*CONFIDENCE_SCORE*/**, for example for the first *sbatch* the result will be in the folder: **Results/Task_1.1/faster_RCNN/0.5/**.
