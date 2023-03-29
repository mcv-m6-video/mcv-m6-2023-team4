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

See the annotations.ipynb notebook

## Task 1.3

Run the file that fine-tunes and then evaluates the network you specify using the parameter *--network*. You can also specify the initial learning rate using the parameter *--lr*. The default one is 1e-3:

```cmd
python task_1.3.py --network faster_RCNN --lr 1e-3
```

## Task 1.4

Run the file that fine-tunes and then evaluates the network you specify using the parameter *--network* using cross-validation. You can also specify the initial learning rate using the parameter *--lr*. The default one is 1e-3. For cross-validation you have to use parameter *--cross-validation--*, there are several options such as None, Normal and Random

```cmd
python task_1.4.py --network faster_RCNN --lr 1e-3 --cross-validation=Normal
```
