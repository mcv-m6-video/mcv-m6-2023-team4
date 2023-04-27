# How to run
This folder contains scripts to compute tracking results on a multi-camera setting. Two approaches relying on metric learning have been implemented. Both of them require the tracking and detection output results of Week 4 as an input.

## Approach 1
This task can be found in the notebook triplet_method_1.ipynb. This approach clusters ids by associating the bboxes that have similar embeddings.

## Approach 2
This task can be found in the notebook triplet_method_2.ipynb. This approach relates tracking ids from different cameras previously obtained with overlap tracking and relates them by comparing their embeddings.

## Metrics
HOTA metrics can be computed by running the trackelval.ipynb notebook.

## Train model on Stanford cars dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
To train model on Stanford cars dataset run triplet_cars_stanford.py
