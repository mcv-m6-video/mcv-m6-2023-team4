# Week 2

The goals of this week are:

Perform background estimation

- Model the background pixels of a video with a simple statistical model that classifies pixels into background / foreground
    - Single per-pixel Gaussian
    - Adaptive / Non-adaptive modeling
- Compare simple models with more complex ones

### Task 1: Implement and evaluate per-pixel Gaussian distribution
- A Gaussian distribution models the background at each pixel
    - First 25% of the video sequence to extract statistics (mean and variance of pixels)
    - Second 75% to segment the foreground and evaluate
- Evaluate results
### Task 2: Implement and evaluate Adaptive Gaussian modeling
- Recursive formulation as moving average
    - First 25% of frames for training
    - Second 75% of frames for background adaptation
- Best pair of values (𝛼, ⍴) to maximize mAP
    - Best 𝛼 for non-recursive case, best ⍴ for recursive case
    - Joint grid/random search over (𝛼, ⍴)
- Compare the adaptive and non-adaptive versions qualitatively and quantitatively
### Task 3: Compare with SOTA and evaluate one SOTA method
- Choose one SOTA method, implement it or find an existing implementation (chosen methods):

### Task 4: Update simple Gaussian model to support color sequences
- Use multiple gaussians in different color spaces

## Execution
Execute the different notebooks